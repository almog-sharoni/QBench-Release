import torch
import torch.nn as nn

try:
    from ..registry.op_registry import OpRegistry
    from ..ops.quant_base import quantize_tensor
    from ..ops.quant_base import _quantize_tensor_cuda
except ImportError:
    from src.registry.op_registry import OpRegistry
    from src.ops.quant_base import quantize_tensor
    from src.ops.quant_base import _quantize_tensor_cuda


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
    _ATTENTION_TYPES = (nn.MultiheadAttention,)
    _FUNCTIONAL_OP_NAMES = (
        "QuantMatMul",
        "QuantBMM",
        "QuantAdd",
        "QuantSub",
        "QuantMul",
        "QuantDiv",
        "QuantCat",
    )

    def __init__(
        self,
        model,
        metric='mse',
        chunk_size=128,
        candidate_formats=None,
        restrict_post_relu_ufp=False,
        unsigned_input_sources=None,
        use_unsigned_input_candidates=True,
    ):
        self.model = model
        self.metric = metric
        self.chunk_size = chunk_size
        self.restrict_post_relu_ufp = bool(restrict_post_relu_ufp)
        self.use_unsigned_input_candidates = bool(use_unsigned_input_candidates)
        self.unsigned_input_sources = {
            str(source).lower()
            for source in (unsigned_input_sources or [])
            if str(source).strip()
        }
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
        self.input_hook_ops = tuple(dict.fromkeys(self.hookable_ops + self._COMPUTE_TYPES + self._ATTENTION_TYPES))
        self.post_relu_layers = self._find_post_relu_layers()
        self.unsigned_passthrough_layers = set()
        self.post_unsigned_layers = self._find_post_unsigned_layers()

        self.ufp_candidates = [f for f in self.candidate_formats if f.startswith('ufp')]
        self.non_ufp_candidates = [f for f in self.candidate_formats if not f.startswith('ufp')]
        self.unsigned_candidate_formats = self._make_unsigned_candidates(self.candidate_formats)

    @staticmethod
    def _dedupe_formats(formats):
        seen = set()
        deduped = []
        for fmt in formats:
            if fmt in seen:
                continue
            seen.add(fmt)
            deduped.append(fmt)
        return deduped

    @staticmethod
    def _to_unsigned_format(fmt):
        if not isinstance(fmt, str) or fmt == 'fp32' or fmt.startswith(('ufp', 'uefp')):
            return fmt

        try:
            from src.ops.quant_softmax import qtype_to_unsigned_qtype
        except ImportError:
            from ..ops.quant_softmax import qtype_to_unsigned_qtype

        return qtype_to_unsigned_qtype(fmt, add_to_mant=True)

    @classmethod
    def _make_unsigned_candidates(cls, candidates):
        return cls._dedupe_formats([cls._to_unsigned_format(fmt) for fmt in candidates])

    @staticmethod
    def _is_unsigned_format(fmt):
        return isinstance(fmt, str) and fmt.startswith(('ufp', 'uefp'))

    def _module_uses_unsigned_input(self, module):
        if not self.use_unsigned_input_candidates:
            return False

        input_q_type = getattr(module, 'input_q_type', None)
        if self._is_unsigned_format(input_q_type):
            return True

        for attr_name in dir(module):
            if not attr_name.startswith('input') or not attr_name.endswith('_q_type'):
                continue
            if self._is_unsigned_format(getattr(module, attr_name, None)):
                return True

        return False

    def _is_unsigned_source_module(self, module):
        if not self.use_unsigned_input_candidates or not self.unsigned_input_sources:
            return False

        class_name = module.__class__.__name__.lower()
        unquantized_name = class_name.replace('quant', '')
        aliases = {class_name, unquantized_name}
        if unquantized_name.endswith('6'):
            aliases.add(unquantized_name[:-1])
        return bool(aliases & self.unsigned_input_sources)

    @staticmethod
    def _is_passthrough_module(module):
        return isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.Identity))

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

    @staticmethod
    def _node_inputs(node):
        all_input_nodes = getattr(node, 'all_input_nodes', None)
        if all_input_nodes is not None:
            return list(all_input_nodes)

        inputs = []

        def collect(value):
            if isinstance(value, torch.fx.Node):
                inputs.append(value)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    collect(item)
            elif isinstance(value, dict):
                for item in value.values():
                    collect(item)

        collect(node.args)
        collect(node.kwargs)
        return inputs

    def _find_post_unsigned_layers_fx(self):
        try:
            from src.utils.fx_trace_utils import trace_quant_aware
        except ImportError:
            from ..utils.fx_trace_utils import trace_quant_aware

        post_unsigned = set()

        def process_graph_module(gm, prefix=""):
            modified = False
            unsigned_nodes = set()
            modules = dict(gm.named_modules())

            for node in gm.graph.nodes:
                is_unsigned_source = False
                is_passthrough = False
                uses_unsigned_input = False

                if node.op == 'call_module':
                    module = modules.get(node.target)
                    if module is not None:
                        is_unsigned_source = self._is_unsigned_source_module(module)
                        is_passthrough = self._is_passthrough_module(module)
                        uses_unsigned_input = self._module_uses_unsigned_input(module)

                if is_unsigned_source or (is_passthrough and uses_unsigned_input):
                    unsigned_nodes.add(node.name)
                    if is_passthrough:
                        layer_name = f"{prefix}.{node.target}" if prefix else str(node.target)
                        self.unsigned_passthrough_layers.add(layer_name)
                elif is_passthrough:
                    node_inputs = self._node_inputs(node)
                    if node_inputs and any(inp.name in unsigned_nodes for inp in node_inputs):
                        unsigned_nodes.add(node.name)
                        layer_name = f"{prefix}.{node.target}" if prefix else str(node.target)
                        self.unsigned_passthrough_layers.add(layer_name)

            for node in gm.graph.nodes:
                if node.op != 'call_module':
                    continue

                module = modules.get(node.target)
                if module is None or self._is_passthrough_module(module):
                    continue

                is_compute = isinstance(module, self._COMPUTE_TYPES) or isinstance(module, self.hookable_ops)
                if not is_compute:
                    continue

                node_inputs = self._node_inputs(node)
                if self._module_uses_unsigned_input(module) or any(inp.name in unsigned_nodes for inp in node_inputs):
                    layer_name = f"{prefix}.{node.target}" if prefix else str(node.target)
                    post_unsigned.add(layer_name)
                    modified = True

            return modified

        try:
            if isinstance(self.model, torch.fx.GraphModule):
                process_graph_module(self.model)
                return post_unsigned, bool(post_unsigned)

            _, _, gm = trace_quant_aware(self.model)
            process_graph_module(gm)
            return post_unsigned, bool(post_unsigned)
        except Exception:
            pass

        traced_any = False
        for child_name, child in self.model.named_children():
            try:
                _, _, gm = trace_quant_aware(child)
                if process_graph_module(gm, child_name):
                    traced_any = True
            except Exception:
                continue

        return post_unsigned, traced_any

    def _find_post_unsigned_layers(self):
        post_unsigned, traced = self._find_post_unsigned_layers_fx()
        if traced:
            return post_unsigned

        post_unsigned = set()
        prev_was_unsigned = False

        for name, module in self.model.named_modules():
            is_compute = isinstance(module, self._COMPUTE_TYPES) or isinstance(module, self.hookable_ops)
            is_unsigned_source = self._is_unsigned_source_module(module)
            uses_unsigned_input = self._module_uses_unsigned_input(module)

            if is_unsigned_source or (uses_unsigned_input and self._is_passthrough_module(module)):
                prev_was_unsigned = True
                if self._is_passthrough_module(module):
                    self.unsigned_passthrough_layers.add(name)
                continue

            if self._is_passthrough_module(module):
                if prev_was_unsigned:
                    self.unsigned_passthrough_layers.add(name)
                continue
            elif is_compute:
                if prev_was_unsigned or uses_unsigned_input:
                    post_unsigned.add(name)
                prev_was_unsigned = False
            elif not isinstance(
                module,
                (
                    nn.Sequential,
                    nn.ModuleList,
                    nn.BatchNorm2d,
                    nn.BatchNorm1d,
                    nn.AdaptiveAvgPool2d,
                    nn.AvgPool2d,
                    nn.MaxPool2d,
                    nn.Flatten,
                ),
            ):
                prev_was_unsigned = False

        return post_unsigned

    def register_hooks(self):
        count_dynamic = 0
        count_ufp = 0
        count_unsigned_candidate_layers = 0

        for name, module in self.model.named_modules():
            if isinstance(module, self.input_hook_ops):
                hook = self._get_attention_hook(name) if isinstance(module, self._ATTENTION_TYPES) else self._get_hook(name)
                self.hooks.append(module.register_forward_pre_hook(hook))
                self.hooked_modules.append(module)
                uses_unsigned_candidates = (
                    self.use_unsigned_input_candidates
                    and (
                        name in self.post_unsigned_layers
                        or name in self.unsigned_passthrough_layers
                    )
                ) or (
                    self._module_uses_unsigned_input(module)
                )
                if uses_unsigned_candidates:
                    count_unsigned_candidate_layers += 1
                if uses_unsigned_candidates or name in self.post_relu_layers:
                    count_ufp += 1
                else:
                    count_dynamic += 1

        if self.restrict_post_relu_ufp:
            print(
                f"Registered hooks on {count_dynamic + count_ufp} layers: "
                f"{count_ufp} post-ReLU (UFP candidates), "
                f"{count_dynamic} other (non-UFP candidates, metric={self.metric.upper()})."
            )
            if not self.ufp_candidates:
                print("  WARNING: no UFP formats in candidates; post-ReLU layers use non-UFP.")
            if not self.non_ufp_candidates:
                print("  WARNING: no non-UFP formats in candidates; other layers use UFP.")
        else:
            print(
                f"Registered hooks on {count_dynamic + count_ufp} layers: "
                f"all layers use all {len(self.candidate_formats)} candidates "
                f"(metric={self.metric.upper()})."
            )
        if self.use_unsigned_input_candidates:
            print(f"{count_unsigned_candidate_layers} layers are using UFP candidates.")

    def _candidates_for_layer(self, layer_name, module=None):
        if (
            (
                self.use_unsigned_input_candidates
                and (
                    layer_name in self.post_unsigned_layers
                    or layer_name in self.unsigned_passthrough_layers
                )
            )
            or (
                module is not None
                and self._module_uses_unsigned_input(module)
            )
        ):
            return self.unsigned_candidate_formats

        if self.restrict_post_relu_ufp:
            if layer_name in self.post_relu_layers:
                return self.ufp_candidates or self._make_unsigned_candidates(self.non_ufp_candidates)
            return self.non_ufp_candidates or self.ufp_candidates

        return self.candidate_formats

    def _get_hook(self, layer_name):
        def hook_fn(module, args):
            x = args[0]
            if not isinstance(x, torch.Tensor):
                return None

            candidates = self._candidates_for_layer(layer_name, module)

            x_quantized, best_indices = self._quantize_input_tensor(x, layer_name, candidates)
            self._set_module_input_quant_state(module, candidates, best_indices)

            # Return the quantized tensor and all other original arguments to replace the input.
            # This ensures multi-argument ops like QuantAdd(x, other) don't lose 'other'.
            return (x_quantized, *args[1:])

        return hook_fn

    def _get_attention_hook(self, layer_name):
        def input_layer_name(suffix):
            return f"{layer_name}.{suffix}" if layer_name else suffix

        def hook_fn(module, args):
            if len(args) < 3:
                return None

            query, key, value = args[:3]
            if not all(isinstance(t, torch.Tensor) for t in (query, key, value)):
                return None

            candidates = self._candidates_for_layer(layer_name, module)
            q_quantized, q_indices = self._quantize_input_tensor(
                query, input_layer_name("query_input"), candidates
            )
            k_quantized, _ = self._quantize_input_tensor(
                key, input_layer_name("key_input"), candidates
            )
            v_quantized, _ = self._quantize_input_tensor(
                value, input_layer_name("value_input"), candidates
            )
            self._set_module_input_quant_state(module, candidates, q_indices)

            return (q_quantized, k_quantized, v_quantized, *args[3:])

        return hook_fn

    def _set_module_input_quant_state(self, module, candidates, best_indices):
        # Disable a quantized module's built-in input quantization path after the
        # pre-hook has already supplied dynamically quantized inputs.
        module.input_quantization = False
        module.input_mode = 'chunk'
        module.input_chunk_size = self.chunk_size

        # Store candidates and indices without turning the full tensor into a
        # Python list; that would force a large GPU-CPU sync.
        module.input_chunk_candidates = candidates
        module.input_chunk_format_indices = best_indices

        if len(candidates) > 0:
            module.input_q_type = candidates[0]

        module.rounding = 'nearest'

    def _quantize_input_tensor(self, tensor, layer_name, candidates):
        quantized, best_indices = self._select_best_format(tensor, layer_name, candidates)

        with torch.no_grad():
            diff = tensor - quantized
            self.stats['sum_l1_err'] += diff.abs().sum().item()
            self.stats['sum_mse_err'] += diff.pow(2).sum().item()
            self.stats['sum_l1_norm'] += tensor.abs().sum().item()
            self.stats['sum_l2_norm'] += tensor.pow(2).sum().item()

        return quantized, best_indices

    def _select_best_format(self, tensor, layer_name, candidates):
        """
        Optimized format selection:
        - Stacks all candidate quantizations and chooses the best per-chunk on GPU.
        - Returns the quantized tensor and the TENSOR of best indices (no tolist()).
        """
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
                # quantize_tensor is already CUDA-accelerated.
                
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
        
        # Gather the best quantized chunks
        gather_indices = best_indices.view(1, total_chunks, 1).expand(1, total_chunks, chunk_size)
        best_qs = torch.gather(all_qs, 0, gather_indices).squeeze(0)

        q_flat = best_qs.view(batch_size, -1)
        if pad_len > 0:
            q_flat = q_flat[:, :num_elements]

        if tensor.dim() > 1:
            quantized_tensor = q_flat.view_as(tensor)
        else:
            quantized_tensor = q_flat.view(-1).view_as(tensor)

        # Update layer-wise format selection statistics (stay on GPU)
        if layer_name not in self.layer_stats:
            # We initialize a tensor to hold counts on GPU
            self.layer_stats[layer_name] = {
                'format_counts_tensor': torch.zeros(len(candidates), dtype=torch.long, device=device),
                'candidates': candidates
            }
        
        stats = self.layer_stats[layer_name]
        counts = torch.bincount(best_indices, minlength=len(candidates))
        stats['format_counts_tensor'] += counts

        # Track running error for progress/debug
        best_errors = all_errors[best_indices, torch.arange(total_chunks, device=device)]
        self.running_error += best_errors.sum().item()
        self.total_chunks += total_chunks

        return quantized_tensor, best_indices

    def get_final_stats(self):
        norm_l1 = self.stats['sum_l1_err'] / self.stats['sum_l1_norm'] if self.stats['sum_l1_norm'] > 0 else 0.0
        norm_mse = self.stats['sum_mse_err'] / self.stats['sum_l2_norm'] if self.stats['sum_l2_norm'] > 0 else 0.0
        
        # Convert GPU stats to the expected dict format for logging/plotting
        processed_layer_stats = {}
        for layer_name, stats in self.layer_stats.items():
            if 'format_counts_tensor' in stats:
                counts_tensor = stats['format_counts_tensor'].cpu()
                candidates = stats['candidates']
                counts_dict = {}
                for idx, count in enumerate(counts_tensor.tolist()):
                    if count > 0:
                        counts_dict[candidates[idx]] = count
                processed_layer_stats[layer_name] = {
                    'format_counts': counts_dict,
                    'total_chunks': int(counts_tensor.sum().item())
                }
            else:
                processed_layer_stats[layer_name] = stats

        return {
            'norm_l1': norm_l1,
            'norm_mse': norm_mse,
            'total_l1': self.stats['sum_l1_err'],
            'total_mse': self.stats['sum_mse_err'],
            'layer_stats': processed_layer_stats
        }

    def cleanup(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        for module in self.hooked_modules:
            # Clear all optimization-related attributes
            for attr in ('input_chunk_formats', 'input_chunk_candidates', 'input_chunk_format_indices'):
                if hasattr(module, attr):
                    setattr(module, attr, None)
            
            # Restore default quantization state if needed
            # (though normally hooks are removed after the run anyway)
            module.input_quantization = True 

        self.hooked_modules = []
