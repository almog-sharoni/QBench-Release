import torch
import torch.nn as nn

try:
    from ..registry.op_registry import OpRegistry
    from ..ops.quant_base import quantize_tensor
except ImportError:
    from src.registry.op_registry import OpRegistry
    from src.ops.quant_base import quantize_tensor


DEFAULT_DYNAMIC_INPUT_CANDIDATES = [
    'fp2_e1m0', 'fp3_e1m1', 'fp4_e1m2', 'fp5_e1m3', 'fp6_e1m4', 'fp7_e1m5', 'fp8_e1m6',
    'fp3_e2m0', 'fp4_e3m0', 'fp5_e4m0', 'fp6_e5m0', 'fp7_e6m0', 'fp8_e7m0'
]


class DynamicInputQuantizer:
    """
    Runtime dynamic input quantizer.

    Registers forward pre-hooks on quantized modules and selects per-chunk
    input formats by minimizing the configured error metric.
    """
    _RELU_TYPES = (nn.ReLU, nn.ReLU6)
    _COMPUTE_TYPES = (nn.Conv2d, nn.Linear)
    _FUNCTIONAL_OP_NAMES = (
        "QuantMatMul",
        "QuantBMM",
        "QuantAdd",
        "QuantSub",
        "QuantMul",
        "QuantDiv",
        "QuantCat",
    )

    def __init__(self, model, metric='mse', chunk_size=128, candidate_formats=None):
        self.model = model
        self.metric = metric
        self.chunk_size = chunk_size
        self.hooks = []
        self.hooked_modules = []
        self.layer_stats = {}
        self.running_error = 0.0
        self.total_chunks = 0

        self.stats = {
            'sum_l1_err': 0.0,
            'sum_mse_err': 0.0,
            'sum_l1_norm': 0.0,
            'sum_l2_norm': 0.0
        }

        if candidate_formats is None:
            candidate_formats = DEFAULT_DYNAMIC_INPUT_CANDIDATES
        self.candidate_formats = list(candidate_formats)

        self.supported_ops = tuple(OpRegistry.get_supported_ops().values())
        functional_ops = []
        for op_name in self._FUNCTIONAL_OP_NAMES:
            try:
                functional_ops.append(OpRegistry.get(op_name))
            except Exception:
                continue
        self.functional_ops = tuple(functional_ops)
        self.hookable_ops = tuple(dict.fromkeys(self.supported_ops + self.functional_ops))
        self.post_relu_layers = self._find_post_relu_layers()

        self.ufp_candidates = [f for f in self.candidate_formats if f.startswith('ufp')]
        self.non_ufp_candidates = [f for f in self.candidate_formats if not f.startswith('ufp')]

    def _find_post_relu_layers(self):
        post_relu = set()
        prev_was_relu = False

        for name, module in self.model.named_modules():
            is_compute = isinstance(module, self._COMPUTE_TYPES) or isinstance(module, self.hookable_ops)
            is_relu = isinstance(module, self._RELU_TYPES)

            if is_compute:
                if prev_was_relu:
                    post_relu.add(name)
                prev_was_relu = False
            elif is_relu:
                prev_was_relu = True
            elif not isinstance(
                module,
                (
                    nn.Sequential,
                    nn.ModuleList,
                    nn.BatchNorm2d,
                    nn.BatchNorm1d,
                    nn.Dropout,
                    nn.Identity,
                    nn.AdaptiveAvgPool2d,
                    nn.AvgPool2d,
                    nn.MaxPool2d,
                    nn.Flatten,
                ),
            ):
                prev_was_relu = False

        return post_relu

    def register_hooks(self):
        count_dynamic = 0
        count_ufp = 0

        for name, module in self.model.named_modules():
            if isinstance(module, self.hookable_ops):
                hook = self._get_hook(name)
                self.hooks.append(module.register_forward_pre_hook(hook))
                self.hooked_modules.append(module)
                if name in self.post_relu_layers:
                    count_ufp += 1
                else:
                    count_dynamic += 1

        print(
            f"Registered hooks on {count_dynamic + count_ufp} layers: "
            f"{count_ufp} post-ReLU (UFP candidates), "
            f"{count_dynamic} other (non-UFP candidates, metric={self.metric.upper()})."
        )
        if not self.ufp_candidates:
            print("  WARNING: no UFP formats in candidates; post-ReLU layers use non-UFP.")
        if not self.non_ufp_candidates:
            print("  WARNING: no non-UFP formats in candidates; other layers use UFP.")

    def _get_hook(self, layer_name):
        def hook_fn(module, args):
            x = args[0]
            if not isinstance(x, torch.Tensor):
                return None

            if layer_name in self.post_relu_layers:
                candidates = self.ufp_candidates or self.non_ufp_candidates
            else:
                candidates = self.non_ufp_candidates or self.ufp_candidates

            x_quantized, chunk_formats = self._select_best_format(x, layer_name, candidates)

            module.input_quantization = True
            module.input_mode = 'chunk'
            module.input_chunk_size = self.chunk_size
            module.input_chunk_formats = chunk_formats
            if chunk_formats:
                module.input_q_type = chunk_formats[0]
            module.rounding = 'nearest'

            with torch.no_grad():
                diff = x - x_quantized
                diff_flat = diff.reshape(-1)
                x_flat = x.reshape(-1)

                self.stats['sum_l1_err'] += diff_flat.abs().sum().item()
                self.stats['sum_mse_err'] += diff_flat.pow(2).sum().item()
                self.stats['sum_l1_norm'] += x_flat.abs().sum().item()
                self.stats['sum_l2_norm'] += x_flat.pow(2).sum().item()

            return None

        return hook_fn

    def _select_best_format(self, tensor, layer_name, candidates):
        return self._dynamic_quantize_per_chunk(tensor, layer_name, candidates)

    def _dynamic_quantize_per_chunk(self, tensor, layer_name, candidates):
        chunk_size = self.chunk_size
        device = tensor.device

        if tensor.dim() > 1:
            flat = tensor.flatten(1)
            batch_size = tensor.shape[0]
        else:
            flat = tensor.flatten(0).unsqueeze(0)
            batch_size = 1

        num_elements = flat.shape[-1]
        pad_len = 0
        if num_elements % chunk_size != 0:
            pad_len = chunk_size - (num_elements % chunk_size)
            flat = torch.nn.functional.pad(flat, (0, pad_len))

        num_chunks = flat.shape[-1] // chunk_size
        ref_chunks = flat.view(batch_size, num_chunks, chunk_size).reshape(-1, chunk_size)
        total_chunks = ref_chunks.shape[0]

        candidate_errors = []
        candidate_qs = []

        for fmt in candidates:
            if fmt == 'fp32':
                candidate_errors.append(torch.full((total_chunks,), float('inf'), device=device))
                candidate_qs.append(ref_chunks)
                continue

            try:
                q_tensor, _ = quantize_tensor(
                    tensor,
                    q_type=fmt,
                    mode='chunk',
                    chunk_size=chunk_size,
                    rounding='nearest',
                )
                q_flat = q_tensor.flatten(1) if q_tensor.dim() > 1 else q_tensor.flatten(0).unsqueeze(0)
                if pad_len > 0:
                    q_flat = torch.nn.functional.pad(q_flat, (0, pad_len))
                q_chunks = q_flat.view(batch_size, num_chunks, chunk_size).reshape(-1, chunk_size)

                diff = ref_chunks - q_chunks
                if self.metric == 'l1':
                    err = diff.abs().mean(dim=1)
                else:
                    err = diff.pow(2).mean(dim=1)

                candidate_errors.append(err)
                candidate_qs.append(q_chunks)
            except Exception:
                candidate_errors.append(torch.full((total_chunks,), float('inf'), device=device))
                candidate_qs.append(ref_chunks)

        all_errors = torch.stack(candidate_errors, dim=0)
        best_indices = torch.argmin(all_errors, dim=0)
        all_qs = torch.stack(candidate_qs, dim=0)
        gather_indices = best_indices.view(1, total_chunks, 1).expand(1, total_chunks, chunk_size)
        best_qs = torch.gather(all_qs, 0, gather_indices).squeeze(0)

        q_flat = best_qs.view(batch_size, -1)
        if pad_len > 0:
            q_flat = q_flat[:, :num_elements]

        if tensor.dim() > 1:
            quantized_tensor = q_flat.view_as(tensor)
        else:
            quantized_tensor = q_flat.view(-1).view_as(tensor)

        if layer_name not in self.layer_stats:
            self.layer_stats[layer_name] = {'format_counts': {}}
        counts_dict = self.layer_stats[layer_name]['format_counts']

        counts = torch.bincount(best_indices, minlength=len(candidates))
        for idx, count in enumerate(counts.cpu().tolist()):
            if count > 0:
                fmt = candidates[idx]
                counts_dict[fmt] = counts_dict.get(fmt, 0) + count

        best_errors = all_errors[best_indices, torch.arange(total_chunks, device=device)]
        self.running_error += best_errors.sum().item()
        self.total_chunks += total_chunks

        best_indices_list = best_indices.detach().cpu().tolist()
        selected_chunk_formats = [candidates[idx] for idx in best_indices_list]

        return quantized_tensor, selected_chunk_formats

    def get_final_stats(self):
        norm_l1 = self.stats['sum_l1_err'] / self.stats['sum_l1_norm'] if self.stats['sum_l1_norm'] > 0 else 0.0
        norm_mse = self.stats['sum_mse_err'] / self.stats['sum_l2_norm'] if self.stats['sum_l2_norm'] > 0 else 0.0
        return {
            'norm_l1': norm_l1,
            'norm_mse': norm_mse,
            'total_l1': self.stats['sum_l1_err'],
            'total_mse': self.stats['sum_mse_err'],
        }

    def cleanup(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        for module in self.hooked_modules:
            if hasattr(module, 'input_chunk_formats'):
                module.input_chunk_formats = None
        self.hooked_modules = []
