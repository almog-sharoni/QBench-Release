import torch
import torch.nn as nn

from src.registry.op_registry import OpRegistry
from src.ops.quant_base import quantize_tensor


class UniformInputQuantizer:
    """
    Applies one fixed quantization format to layer inputs through pre-hooks.
    Captures quantization error statistics for reporting/logging.
    """
    _FUNCTIONAL_OP_NAMES = (
        "QuantMatMul",
        "QuantBMM",
        "QuantAdd",
        "QuantSub",
        "QuantMul",
        "QuantDiv",
        "QuantCat",
    )

    def __init__(self, model, fmt, chunk_size=128):
        self.model = model
        self.fmt = fmt
        self.chunk_size = chunk_size
        self.hooks = []
        self.layer_stats = {}
        self.supported_ops = tuple(OpRegistry.get_supported_ops().values())
        functional_ops = []
        for op_name in self._FUNCTIONAL_OP_NAMES:
            try:
                functional_ops.append(OpRegistry.get(op_name))
            except Exception:
                continue
        self.functional_ops = tuple(functional_ops)
        self.hookable_ops = tuple(dict.fromkeys(self.supported_ops + self.functional_ops))
        self.stats = {
            'sum_l1_err': 0.0,
            'sum_mse_err': 0.0,
            'sum_l1_norm': 0.0,
            'sum_l2_norm': 0.0,
        }

    def _quantize(self, x):
        x_q, _ = quantize_tensor(
            x,
            q_type=self.fmt,
            mode='chunk',
            chunk_size=self.chunk_size,
            rounding='nearest',
        )
        return x_q

    def _make_hook(self, layer_name):
        def hook(module, inputs):
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return None

            x = inputs[0]
            x_q = self._quantize(x)

            if x.dim() > 1:
                flat = x.flatten(1)
                batch_size = x.shape[0]
            else:
                flat = x.flatten(0).unsqueeze(0)
                batch_size = 1

            num_elements = flat.shape[-1]
            pad_len = 0
            if num_elements % self.chunk_size != 0:
                pad_len = self.chunk_size - (num_elements % self.chunk_size)
                num_elements += pad_len
            num_chunks = num_elements // self.chunk_size
            total_chunks = batch_size * num_chunks

            module.input_quantization = True
            module.input_mode = 'chunk'
            module.input_chunk_size = self.chunk_size
            module.input_q_type = self.fmt
            module.input_chunk_formats = None
            module.rounding = 'nearest'

            with torch.no_grad():
                diff = x - x_q
                self.stats['sum_l1_err'] += diff.abs().sum().item()
                self.stats['sum_mse_err'] += diff.pow(2).sum().item()
                self.stats['sum_l1_norm'] += x.abs().sum().item()
                self.stats['sum_l2_norm'] += x.pow(2).sum().item()

            stats = self.layer_stats.setdefault(
                layer_name,
                {'format_counts': {}, 'total_chunks': 0, 'type': module.__class__.__name__}
            )
            stats['format_counts'][self.fmt] = stats['format_counts'].get(self.fmt, 0) + total_chunks
            stats['total_chunks'] += total_chunks

            return None

        return hook

    def register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, self.hookable_ops) or isinstance(module, (nn.Conv2d, nn.Linear)):
                self.hooks.append(module.register_forward_pre_hook(self._make_hook(name)))

    def cleanup(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_final_stats(self):
        norm_l1 = self.stats['sum_l1_err'] / self.stats['sum_l1_norm'] if self.stats['sum_l1_norm'] > 0 else 0.0
        norm_mse = self.stats['sum_mse_err'] / self.stats['sum_l2_norm'] if self.stats['sum_l2_norm'] > 0 else 0.0
        return {
            'norm_l1': norm_l1,
            'norm_mse': norm_mse,
            'total_l1': self.stats['sum_l1_err'],
            'total_mse': self.stats['sum_mse_err'],
        }
