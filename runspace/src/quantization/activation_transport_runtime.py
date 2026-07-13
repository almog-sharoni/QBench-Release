"""FX graph instrumentation for hardware activation transport."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import types
import weakref

import torch
import torch.nn as nn

from .activation_stage_planner import ActivationStagePlan, plan_activation_stages
from .activation_transport import ActivationPacket, ActivationTransport


_MISSING = object()


@dataclass(frozen=True)
class ActivationBypass:
    """Explicitly carry an unquantized stage without counting a transmission."""

    value: torch.Tensor


@dataclass
class _TransportValue:
    value: ActivationPacket | torch.Tensor
    transmission_id: int
    observed: bool = False
    decoded: torch.Tensor | None = None


class _StageEncoder(nn.Module):
    def __init__(self, runtime, stage_id: str):
        super().__init__()
        object.__setattr__(self, "_runtime_ref", weakref.ref(runtime))
        self.stage_id = str(stage_id)

    def forward(self, value):
        runtime = self._runtime_ref()
        if runtime is None:
            raise RuntimeError("Activation transport runtime was released during forward")
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Activation stage {self.stage_id!r} produced "
                f"{type(value).__name__}; expected a Tensor"
            )
        return runtime.encode_stage(self.stage_id, value)


class _StageDecoder(nn.Module):
    def __init__(self, runtime, stage_id: str, consumer_name: str):
        super().__init__()
        object.__setattr__(self, "_runtime_ref", weakref.ref(runtime))
        self.stage_id = str(stage_id)
        self.consumer_name = str(consumer_name)

    def forward(self, value):
        runtime = self._runtime_ref()
        if runtime is None:
            raise RuntimeError("Activation transport runtime was released during forward")
        if isinstance(value, ActivationBypass):
            return value.value
        if not isinstance(value, _TransportValue):
            raise TypeError(
                "Activation stage decoder expected an internal transport value; "
                f"got {type(value).__name__}"
            )
        runtime.decode_reads += 1
        decoded = runtime.decode_stage(value, self.stage_id)
        runtime.observe_decode(
            value,
            self.stage_id,
            decoded,
        )
        return decoded


def _module_name(prefix: str, stage_id: str, suffix: str = "") -> str:
    safe = "".join(char if char.isalnum() or char == "_" else "_" for char in stage_id)
    tail = f"_{suffix}" if suffix else ""
    return f"_qbench_{prefix}_{safe}{tail}"


class ActivationTransportRuntime:
    """Instrument one model with explicit stage-output encode/decode nodes.

    ``encode_callback`` receives ``(stage, tensor)`` and returns an
    ``ActivationPacket`` (encoded mode), decoded Tensor (reference mode), or
    ``ActivationBypass``. ``decode_observer`` sees the first decoded consumer
    value for each transmission, including fan-out. The same stage planner and
    graph placement are used in both modes.
    """

    def __init__(
        self,
        model: nn.Module,
        transport: ActivationTransport,
        encode_callback,
        *,
        planner_kwargs=None,
        decode_observer=None,
    ):
        self.model = model
        self.transport = transport
        self.encode_callback = encode_callback
        self.decode_observer = decode_observer
        self.planner_kwargs = dict(planner_kwargs or {})
        self.plan: ActivationStagePlan | None = None
        self.graph_module: torch.fx.GraphModule | None = None
        self.transmission_count = 0
        self.packet_count = 0
        self.decode_reads = 0
        self.encoded_bytes = 0
        self.stage_transmissions = {}
        self.stage_encoded_bytes = {}
        self._installed = False
        self._original_forward = None
        self._original_quant_flags = {}
        self._next_transmission_id = 0
        self._metadata_bypass_stages = {}
        self._stages_by_id = {}

    def _stage_map(self):
        return self._stages_by_id

    def decode_stage(
        self,
        transmitted: _TransportValue,
        stage_id: str,
    ) -> torch.Tensor:
        value = transmitted.value
        if not isinstance(value, ActivationPacket):
            return self.transport.decode(value)

        stage = self._stages_by_id.get(stage_id)
        if stage is None or not stage.has_fanout:
            return self.transport.decode(value)

        # Encoded fan-out previously decoded each consumer into distinct
        # storage. Decode once, then clone the cached value so in-place users
        # retain the same aliasing behavior without repeating the codec work.
        if transmitted.decoded is None:
            transmitted.decoded = self.transport.decode(value)
        return transmitted.decoded.clone()

    def encode_stage(self, stage_id: str, tensor: torch.Tensor):
        stage = self._stage_map().get(stage_id)
        if stage is None:
            raise KeyError(f"Unknown activation stage {stage_id!r}")
        if stage_id in self._metadata_bypass_stages or not tensor.dtype.is_floating_point:
            return ActivationBypass(tensor)
        transmitted = self.encode_callback(stage, tensor)
        if isinstance(transmitted, ActivationBypass):
            if not isinstance(transmitted.value, torch.Tensor):
                raise TypeError(
                    "ActivationBypass.value must be a Tensor; got "
                    f"{type(transmitted.value).__name__} for stage {stage_id!r}"
                )
            return transmitted
        self.transmission_count += 1
        self.stage_transmissions[stage_id] = self.stage_transmissions.get(stage_id, 0) + 1
        if isinstance(transmitted, ActivationPacket):
            self.packet_count += 1
            self.encoded_bytes += transmitted.encoded_nbytes
            self.stage_encoded_bytes[stage_id] = (
                self.stage_encoded_bytes.get(stage_id, 0) + transmitted.encoded_nbytes
            )
        elif not isinstance(transmitted, torch.Tensor):
            raise TypeError(
                "Activation transport encoder must return ActivationPacket or Tensor; "
                f"got {type(transmitted).__name__} for stage {stage_id!r}"
            )
        transmission_id = self._next_transmission_id
        self._next_transmission_id += 1
        return _TransportValue(transmitted, transmission_id)

    def observe_decode(
        self,
        transmitted: _TransportValue,
        stage_id: str,
        decoded: torch.Tensor,
    ) -> None:
        if transmitted.observed:
            return
        transmitted.observed = True
        if self.decode_observer is not None:
            self.decode_observer(stage_id, decoded)

    @staticmethod
    def _node_by_name(graph):
        return {node.name: node for node in graph.nodes}

    def _instrument(self, plan: ActivationStagePlan) -> torch.fx.GraphModule:
        graph_module = plan.graph_module
        graph = graph_module.graph
        nodes = self._node_by_name(graph)

        for stage in plan.stages:
            output_name = stage.output_node
            if not output_name or output_name not in nodes:
                continue
            output_node = nodes[output_name]
            if output_node.op in {"output", "get_attr"}:
                continue

            original_users = list(output_node.users)
            if not original_users:
                continue

            encoder_name = _module_name("encode", stage.stage_id)
            index = 1
            while hasattr(graph_module, encoder_name):
                encoder_name = _module_name("encode", stage.stage_id, str(index))
                index += 1
            graph_module.add_submodule(encoder_name, _StageEncoder(self, stage.stage_id))
            with graph.inserting_after(output_node):
                encoded_node = graph.call_module(encoder_name, args=(output_node,))

            for user_index, user in enumerate(original_users):
                decoder_name = _module_name(
                    "decode",
                    stage.stage_id,
                    f"{user.name}_{user_index}",
                )
                suffix = 1
                base_name = decoder_name
                while hasattr(graph_module, decoder_name):
                    decoder_name = f"{base_name}_{suffix}"
                    suffix += 1
                graph_module.add_submodule(
                    decoder_name,
                    _StageDecoder(self, stage.stage_id, user.name),
                )
                with graph.inserting_before(user):
                    decoded_node = graph.call_module(decoder_name, args=(encoded_node,))
                user.replace_input_with(output_node, decoded_node)

        graph.lint()
        graph_module.recompile()
        graph_module.train(self.model.training)
        return graph_module

    def _disable_module_boundary_quantization(self):
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

    def install(self):
        if self._installed:
            return self
        planning_model = self.model
        planner_kwargs = dict(self.planner_kwargs)
        trace_provider = getattr(
            self.model,
            "_qbench_activation_trace_provider",
            None,
        )
        if callable(trace_provider):
            provided = trace_provider()
            provider_kwargs = {}
            if isinstance(provided, tuple):
                if len(provided) != 2:
                    raise TypeError(
                        "Activation trace provider must return GraphModule or "
                        "(GraphModule, planner_kwargs)"
                    )
                planning_model, provider_kwargs = provided
            else:
                planning_model = provided
            if not isinstance(planning_model, torch.fx.GraphModule):
                raise TypeError(
                    "Activation trace provider must return a torch.fx.GraphModule; "
                    f"got {type(planning_model).__name__}"
                )
            if not isinstance(provider_kwargs, dict):
                raise TypeError("Activation trace provider planner_kwargs must be a mapping")
            planner_kwargs = {**provider_kwargs, **planner_kwargs}
        elif isinstance(self.model, torch.fx.GraphModule):
            # Instrument a separate GraphModule that shares the original modules.
            # Calling an instrumented original from its patched forward would recurse.
            planning_model = torch.fx.GraphModule(
                self.model,
                copy.deepcopy(self.model.graph),
            )
        self.plan = plan_activation_stages(planning_model, **planner_kwargs)
        self._stages_by_id = {
            stage.stage_id: stage for stage in self.plan.stages
        }
        planned_nodes = {
            node.name: node for node in self.plan.graph_module.graph.nodes
        }
        self._metadata_bypass_stages = {
            stage.stage_id: planned_nodes[stage.output_node].meta[
                "qbench_activation_bypass"
            ]
            for stage in self.plan.stages
            if stage.output_node in planned_nodes
            and planned_nodes[stage.output_node].meta.get("qbench_activation_bypass")
        }
        self.graph_module = self._instrument(self.plan)
        self._disable_module_boundary_quantization()
        self._original_forward = self.model.forward
        runtime_ref = weakref.ref(self)

        def transport_forward(_model, *args, **kwargs):
            runtime = runtime_ref()
            if runtime is None or runtime.graph_module is None:
                raise RuntimeError("Activation transport runtime is not installed")
            return runtime.graph_module(*args, **kwargs)

        self.model.forward = types.MethodType(transport_forward, self.model)
        self._installed = True
        return self

    def cleanup(self):
        if not self._installed:
            return
        if self._original_forward is not None:
            self.model.forward = self._original_forward
        for module, saved in self._original_quant_flags.values():
            for attr, value in saved.items():
                if value is _MISSING:
                    delattr(module, attr)
                else:
                    setattr(module, attr, value)
        self._original_quant_flags.clear()
        self.graph_module = None
        self.plan = None
        self._metadata_bypass_stages.clear()
        self._stages_by_id.clear()
        self._installed = False

    def transport_stats(self):
        stages = self.plan.stages if self.plan is not None else ()
        return {
            "transport": self.transport.mode,
            "transmission_count": int(self.transmission_count),
            "packet_count": int(self.packet_count),
            "decode_reads": int(self.decode_reads),
            "encoded_bytes": int(self.encoded_bytes),
            "planner_version": 1,
            "stage_count": len(stages),
            "unsigned_stage_count": sum(stage.is_unsigned for stage in stages),
            "activation_plan": {
                stage.stage_id: {
                    "kind": stage.kind.value,
                    "producer_nodes": list(stage.node_names),
                    "consumer_nodes": list(stage.consumer_nodes),
                    "consumer_stage_ids": list(stage.consumer_stage_ids),
                    "is_unsigned": bool(stage.is_unsigned),
                    "unsigned_source": stage.unsigned_source,
                    "has_fanout": bool(stage.has_fanout),
                    "transmissions": int(
                        self.stage_transmissions.get(stage.stage_id, 0)
                    ),
                    "encoded_bytes": int(
                        self.stage_encoded_bytes.get(stage.stage_id, 0)
                    ),
                    "bypass_reason": self._metadata_bypass_stages.get(
                        stage.stage_id
                    ),
                }
                for stage in stages
            },
        }


__all__ = ["ActivationBypass", "ActivationTransportRuntime"]
