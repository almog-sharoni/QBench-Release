import torch
import torch.nn as nn

from src.registry.op_registry import OpRegistry
from src.ops.quant_base import quantize_tensor


class UniformInputQuantizer:
    """
    Applies one fixed quantization format to layer inputs through pre-hooks.
    Captures quantization error statistics for reporting/logging.
    """
    def __init__(self, model, fmt, chunk_size=128):
        self.model = model
        self.fmt = fmt
        self.chunk_size = chunk_size
        self.hooks = []
        self.layer_names = []
        self.supported_ops = tuple(OpRegistry.get_supported_ops().values())
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
                return inputs

            x = inputs[0]
            x_q = self._quantize(x)

            with torch.no_grad():
                diff = x - x_q
                self.stats['sum_l1_err'] += diff.abs().sum().item()
                self.stats['sum_mse_err'] += diff.pow(2).sum().item()
                self.stats['sum_l1_norm'] += x.abs().sum().item()
                self.stats['sum_l2_norm'] += x.pow(2).sum().item()

            return (x_q,) + inputs[1:]

        return hook

    def register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, self.supported_ops) or isinstance(module, (nn.Conv2d, nn.Linear)):
                self.hooks.append(module.register_forward_pre_hook(self._make_hook(name)))
                self.layer_names.append(name)

    def cleanup(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    @property
    def layer_stats(self):
        return {
            layer_name: {'format_counts': {self.fmt: 1}, 'total_chunks': 1}
            for layer_name in self.layer_names
        }

    def get_final_stats(self):
        norm_l1 = self.stats['sum_l1_err'] / self.stats['sum_l1_norm'] if self.stats['sum_l1_norm'] > 0 else 0.0
        norm_mse = self.stats['sum_mse_err'] / self.stats['sum_l2_norm'] if self.stats['sum_l2_norm'] > 0 else 0.0
        return {
            'norm_l1': norm_l1,
            'norm_mse': norm_mse,
            'total_l1': self.stats['sum_l1_err'],
            'total_mse': self.stats['sum_mse_err'],
        }
