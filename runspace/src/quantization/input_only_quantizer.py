"""Model-input-only quantization through activation transport packets."""

from __future__ import annotations

import types
import weakref

import torch

from .activation_transport import (
    ActivationPacket,
    ActivationTransport,
    normalize_activation_transport,
)
from .chunking import count_context_chunks


_MISSING = object()


class InputOnlyActivationQuantizer:
    """Quantize only the model input while using the hardware transport codec."""

    def __init__(
        self,
        *,
        model,
        fmt: str,
        chunk_size: int = 128,
        transport: str = "encoded",
        collect_error_stats: bool = True,
    ) -> None:
        if not fmt:
            raise ValueError("Input-only activation quantization requires a format")
        self.fmt = str(fmt)
        self.model = model
        self.chunk_size = int(chunk_size)
        self.transport = normalize_activation_transport(transport)
        self.collect_error_stats = bool(collect_error_stats)
        self.activation_transport = ActivationTransport(
            mode=self.transport,
            chunk_size=self.chunk_size,
        )
        self.transmission_count = 0
        self.packet_count = 0
        self.decode_reads = 0
        self.encoded_bytes = 0
        self.total_chunks = 0
        self.sum_l1_err = None
        self.sum_mse_err = None
        self.sum_l1_norm = None
        self.sum_l2_norm = None
        self.last_quantized_input = None
        self._original_forward = None
        self._original_quant_flags = {}
        self._installed = False

    def _disable_module_boundary_quantization(self) -> None:
        for module in self.model.modules():
            saved = {
                "_qbench_activation_transport_active": getattr(
                    module,
                    "_qbench_activation_transport_active",
                    _MISSING,
                )
            }
            module._qbench_activation_transport_active = True
            for attr in ("input_quantization", "output_quantization"):
                if hasattr(module, attr):
                    saved[attr] = getattr(module, attr)
                    setattr(module, attr, False)
            self._original_quant_flags[id(module)] = (module, saved)

    def register_hooks(self) -> None:
        if self._installed:
            return
        self._original_forward = self.model.forward
        quantizer_ref = weakref.ref(self)

        def input_transport_forward(_model, *args, **kwargs):
            quantizer = quantizer_ref()
            if quantizer is None or quantizer._original_forward is None:
                raise RuntimeError("Input-only activation transport is not installed")
            if args:
                quantized_input = quantizer.quantize(args[0])
                return quantizer._original_forward(
                    quantized_input,
                    *args[1:],
                    **kwargs,
                )
            quantized_kwargs = quantizer.quantize(kwargs)
            return quantizer._original_forward(**quantized_kwargs)

        self.model.forward = types.MethodType(input_transport_forward, self.model)
        self._disable_module_boundary_quantization()
        self._installed = True

    def _quantize_tensor(self, tensor: torch.Tensor, producer_id: str) -> torch.Tensor:
        if not tensor.dtype.is_floating_point:
            return tensor
        transmitted = self.activation_transport.transmit_uniform(
            tensor,
            self.fmt,
            producer_id=producer_id,
        )
        quantized = self.activation_transport.decode(transmitted)
        chunks = (
            transmitted.num_chunks
            if isinstance(transmitted, ActivationPacket)
            else count_context_chunks(tensor, self.chunk_size)
        )

        self.transmission_count += 1
        self.decode_reads += 1
        self.total_chunks += chunks
        if isinstance(transmitted, ActivationPacket):
            self.packet_count += 1
            self.encoded_bytes += transmitted.encoded_nbytes

        if self.collect_error_stats:
            with torch.no_grad():
                diff = tensor - quantized
                updates = {
                    "sum_l1_err": diff.abs().sum(),
                    "sum_mse_err": diff.square().sum(),
                    "sum_l1_norm": tensor.abs().sum(),
                    "sum_l2_norm": tensor.square().sum(),
                }
                for attr, value in updates.items():
                    value = value.detach().to(dtype=torch.float64)
                    current = getattr(self, attr)
                    if current is None:
                        setattr(self, attr, value)
                    else:
                        current.add_(value)
        return quantized

    def _quantize_value(self, value, producer_id: str):
        if isinstance(value, torch.Tensor):
            return self._quantize_tensor(value, producer_id)
        if isinstance(value, dict):
            return {
                key: self._quantize_value(item, f"{producer_id}.{key}")
                for key, item in value.items()
            }
        if isinstance(value, tuple):
            return tuple(
                self._quantize_value(item, f"{producer_id}.{index}")
                for index, item in enumerate(value)
            )
        if isinstance(value, list):
            return [
                self._quantize_value(item, f"{producer_id}.{index}")
                for index, item in enumerate(value)
            ]
        return value

    def quantize(self, value):
        quantized = self._quantize_value(value, "model_input")
        self.last_quantized_input = quantized
        return quantized

    def get_final_stats(self):
        sum_l1_err = float(self.sum_l1_err.item()) if self.sum_l1_err is not None else 0.0
        sum_mse_err = float(self.sum_mse_err.item()) if self.sum_mse_err is not None else 0.0
        sum_l1_norm = float(self.sum_l1_norm.item()) if self.sum_l1_norm is not None else 0.0
        sum_l2_norm = float(self.sum_l2_norm.item()) if self.sum_l2_norm is not None else 0.0
        norm_l1 = sum_l1_err / sum_l1_norm if sum_l1_norm else 0.0
        norm_mse = sum_mse_err / sum_l2_norm if sum_l2_norm else 0.0
        stage = {
            "kind": "input",
            "producer_nodes": ["model_input"],
            "consumer_nodes": ["model"],
            "consumer_stage_ids": [],
            "is_unsigned": False,
            "unsigned_source": None,
            "has_fanout": False,
            "transmissions": int(self.transmission_count),
            "encoded_bytes": int(self.encoded_bytes),
        }
        return {
            "transport": self.transport,
            "transmission_count": int(self.transmission_count),
            "packet_count": int(self.packet_count),
            "decode_reads": int(self.decode_reads),
            "encoded_bytes": int(self.encoded_bytes),
            "planner_version": 1,
            "stage_count": 1,
            "unsigned_stage_count": 0,
            "activation_plan": {"model_input": stage},
            "norm_l1": norm_l1,
            "norm_mse": norm_mse,
            "total_l1": sum_l1_err,
            "total_mse": sum_mse_err,
            "collect_error_stats": self.collect_error_stats,
            "layer_stats": {
                "model_input": {
                    "type": "model_input",
                    "format": self.fmt,
                    "format_counts": {self.fmt: int(self.total_chunks)},
                    "total_chunks": int(self.total_chunks),
                    "producer_nodes": ["model_input"],
                    "consumer_nodes": ["model"],
                    "is_unsigned": False,
                    "unsigned_source": None,
                    "candidate_formats": [self.fmt],
                }
            },
        }

    def cleanup(self) -> None:
        if self._installed and self._original_forward is not None:
            self.model.forward = self._original_forward
        for module, saved in self._original_quant_flags.values():
            for attr, value in saved.items():
                if value is _MISSING:
                    delattr(module, attr)
                else:
                    setattr(module, attr, value)
        self._original_quant_flags.clear()
        self._original_forward = None
        self.last_quantized_input = None
        self._installed = False


__all__ = ["InputOnlyActivationQuantizer"]
