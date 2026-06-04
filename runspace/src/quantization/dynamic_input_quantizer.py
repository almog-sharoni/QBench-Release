import torch
import torch.nn as nn

try:
    from ..registry.op_registry import OpRegistry
    from ..ops.quant_base import quantize_tensor
    from ..ops.quant_base import _quantize_tensor_cuda
    from ..cuda import search_best_chunk_format
    from .constants import get_format_params
    from .chunking import chunk_tensor_by_context, unchunk_tensor_by_context
except ImportError:
    from src.registry.op_registry import OpRegistry
    from src.ops.quant_base import quantize_tensor
    from src.ops.quant_base import _quantize_tensor_cuda
    from src.quantization.cuda import search_best_chunk_format
    from src.quantization.constants import get_format_params
    from src.quantization.chunking import chunk_tensor_by_context, unchunk_tensor_by_context

import os
import json


DEFAULT_DYNAMIC_INPUT_CANDIDATES = [
    'fp2_e1m0', 'fp3_e1m1', 'fp4_e1m2', 'fp5_e1m3', 'fp6_e1m4', 'fp7_e1m5', 'fp8_e1m6',
    'fp3_e2m0', 'fp4_e3m0', 'fp5_e4m0', 'fp6_e5m0', 'fp7_e6m0', 'fp8_e7m0',
    'ufp8_e1m7', 'ufp8_e2m6', 'ufp8_e3m5', 'ufp8_e4m4', 'ufp8_e5m3', 'ufp8_e6m2', 'ufp8_e7m1', 'ufp8_e8m0',
    'ufp7_e1m6', 'ufp7_e2m5', 'ufp7_e3m4', 'ufp7_e4m3', 'ufp7_e5m2', 'ufp7_e6m1', 'ufp7_e7m0',
    'ufp6_e1m5', 'ufp6_e2m4', 'ufp6_e3m3', 'ufp6_e4m2', 'ufp6_e5m1', 'ufp6_e6m0',
    'ufp5_e1m4', 'ufp5_e2m3', 'ufp5_e3m2', 'ufp5_e4m1', 'ufp5_e5m0',
    'ufp4_e1m3', 'ufp4_e2m2', 'ufp4_e3m1', 'ufp4_e4m0',
    'ufp3_e1m2', 'ufp3_e2m1', 'ufp3_e3m0',
    'ufp2_e1m1', 'ufp2_e2m0'
]


class DynamicInputQuantizer:
    """
    Runtime dynamic input quantizer.

    Registers forward pre-hooks on quantized modules and selects per-chunk
    input formats by minimizing MSE.
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
        use_cache_sim_db=False,
        model_name=None,
        skip_depthwise_input_quant=False,
        input_transfer_map=None,
        collect_error_stats=True,
        collect_format_stats=True,
    ):
        self.model = model
        self.metric = 'mse'
        self.chunk_size = chunk_size
        self.restrict_post_relu_ufp = bool(restrict_post_relu_ufp)
        self.use_unsigned_input_candidates = bool(use_unsigned_input_candidates)
        self.skip_depthwise_input_quant = bool(skip_depthwise_input_quant)
        self.collect_error_stats = bool(collect_error_stats)
        self.collect_format_stats = bool(collect_format_stats)
        self._candidate_param_cache = {}
        self.unsigned_input_sources = {
            str(source).lower()
            for source in (unsigned_input_sources or [])
            if str(source).strip()
        }
        self.hooks = []
        self.hooked_modules = []
        self.skipped_depthwise_modules = []
        self.layer_stats = {}
        self.running_error = 0.0
        self.total_chunks = 0
        self.layer_unsigned_input_indices = {}

        self.stats = {
            'sum_mse_err': None,
            'sum_l2_norm': None
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

        self.cache_sim_map = {}
        self.layer_residual_input_bits_map = {}
        self.layer_need_input_transfer_map = dict(input_transfer_map or {})
        if use_cache_sim_db and model_name:
            print(f"Loading cache simulation results from DB for {model_name}")
            try:
                try:
                    from src.database.handler import RunDatabase
                except ImportError:
                    from ..database.handler import RunDatabase
                db = RunDatabase()
                sim = db.get_latest_cache_simulation(model_name)
                if sim:
                    for layer in sim.get('layers', []):
                        self.cache_sim_map[layer['name']] = layer.get('stay_on_chip', True)
                        residual_bits = layer.get('residual_input_bits')
                        if residual_bits is not None:
                            self.layer_residual_input_bits_map[layer['name']] = int(residual_bits)
                        need_input_transfer = layer.get('need_input_transfer')
                        if need_input_transfer is not None:
                            self.layer_need_input_transfer_map[layer['name']] = bool(need_input_transfer)
                    print(f"Loaded {len(self.cache_sim_map)} layer statuses from cache sim DB.")
                    if self.cache_sim_map:
                        sample_keys = list(self.cache_sim_map.keys())[:10]
                        print(f"Sample keys in cache sim map: {sample_keys}")
                else:
                    print(f"No cache simulation found in DB for {model_name}.")
            except Exception as e:
                print(f"Failed to fetch cache sim from DB: {e}")
                
        self.all_fp8_formats = ['fp8_e4m3', 'fp8_e5m2', 'fp8_e2m5', 'fp8_e3m4', 'fp8_e1m6', 'fp8_e6m1', 'fp8_e7m0']
        self.unsigned_all_fp8_formats = self._make_unsigned_candidates(self.all_fp8_formats)
        self._reported_missing_layers = set()

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

    def _mark_unsigned_input(self, layer_name, input_index=0):
        self.layer_unsigned_input_indices.setdefault(layer_name, set()).add(int(input_index))

    def _module_unsigned_input_indices(self, module, input_count=None):
        if not self.use_unsigned_input_candidates:
            return set()

        indices = set()
        is_multi_input = self._is_multi_input_module(module)

        if not is_multi_input and self._is_unsigned_format(getattr(module, 'input_q_type', None)):
            indices.add(0)

        for attr_name in dir(module):
            if not attr_name.startswith('input') or not attr_name.endswith('_q_type'):
                continue

            middle = attr_name[len('input'):-len('_q_type')]
            if not middle:
                if not is_multi_input and self._is_unsigned_format(getattr(module, attr_name, None)):
                    indices.add(0)
                continue

            if not middle.isdigit():
                continue

            if self._is_unsigned_format(getattr(module, attr_name, None)):
                indices.add(int(middle) - 1)

        if input_count is not None:
            indices = {idx for idx in indices if 0 <= idx < input_count}

        return indices

    def _module_uses_unsigned_input(self, module):
        return bool(self._module_unsigned_input_indices(module))

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
    def _is_multi_input_module(module):
        class_name = module.__class__.__name__.lower()
        unquantized_name = class_name.replace('quant', '')
        return unquantized_name in ('add', 'mul', 'sub', 'div', 'matmul', 'bmm', 'cat')

    @staticmethod
    def _is_passthrough_module(module):
        return isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.Identity))

    @staticmethod
    def _is_depthwise_conv(module):
        return (
            isinstance(module, nn.Conv2d)
            and getattr(module, 'groups', 1) > 1
            and getattr(module, 'groups', 1) == getattr(module, 'in_channels', None)
            and getattr(module, 'groups', 1) == getattr(module, 'out_channels', None)
        )

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
                static_unsigned_indices = self._module_unsigned_input_indices(
                    module,
                    len(node_inputs),
                )
                if len(node_inputs) > 1:
                    unsigned_input_indices = {
                        idx for idx, inp in enumerate(node_inputs)
                        if inp.name in unsigned_nodes
                    }
                else:
                    unsigned_input_indices = (
                        {0} if any(inp.name in unsigned_nodes for inp in node_inputs) else set()
                    )
                unsigned_input_indices.update(static_unsigned_indices)
                is_post_unsigned = bool(unsigned_input_indices)
                
                if is_post_unsigned:
                    layer_name = f"{prefix}.{node.target}" if prefix else str(node.target)
                    post_unsigned.add(layer_name)
                    for input_index in unsigned_input_indices:
                        self._mark_unsigned_input(layer_name, input_index)
                    modified = True

            return modified

        try:
            if isinstance(self.model, torch.fx.GraphModule):
                process_graph_module(self.model)
                return post_unsigned, True

            _, _, gm = trace_quant_aware(self.model)
            process_graph_module(gm)
            return post_unsigned, True
        except Exception:
            pass

        traced_any = False
        for child_name, child in self.model.named_children():
            try:
                _, _, gm = trace_quant_aware(child)
                traced_any = True
                process_graph_module(gm, child_name)
            except Exception:
                continue

        return post_unsigned, traced_any

    def _find_post_unsigned_layers(self):
        if not self.use_unsigned_input_candidates or not self.unsigned_input_sources:
            return set()

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
                # Conservative fallback: we don't know the exact graph connections here,
                # but if the immediate previous module was unsigned, we assume it's a direct connection.
                # However, for hybrid quant, it's safer to be conservative.
                if prev_was_unsigned and not self._is_multi_input_module(module):
                    post_unsigned.add(name)
                    self._mark_unsigned_input(name, 0)
                elif uses_unsigned_input:
                    post_unsigned.add(name)
                    for input_index in self._module_unsigned_input_indices(module):
                        self._mark_unsigned_input(name, input_index)
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

    def _layer_uses_unsigned_input(self, layer_name, input_index=0):
        if not self.use_unsigned_input_candidates:
            return False

        unsigned_indices = self.layer_unsigned_input_indices.get(layer_name)
        if unsigned_indices:
            return int(input_index) in unsigned_indices

        return (
            int(input_index) == 0
            and (
                layer_name in self.post_unsigned_layers
                or layer_name in self.unsigned_passthrough_layers
            )
        )

    def register_hooks(self):
        count_dynamic = 0
        count_ufp = 0
        count_unsigned_candidate_layers = 0
        count_skipped_depthwise = 0

        for name, module in self.model.named_modules():
            if isinstance(module, self.input_hook_ops):
                if self.skip_depthwise_input_quant and self._is_depthwise_conv(module):
                    module.input_quantization = False
                    module.input_mode = 'fp32'
                    self.skipped_depthwise_modules.append(module)
                    count_skipped_depthwise += 1
                    continue

                hook = self._get_attention_hook(name) if isinstance(module, self._ATTENTION_TYPES) else self._get_hook(name)
                self.hooks.append(module.register_forward_pre_hook(hook))
                self.hooked_modules.append(module)
                uses_unsigned_candidates = (
                    self._layer_uses_unsigned_input(name, input_index=0)
                    or bool(self.layer_unsigned_input_indices.get(name))
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
                f"{count_dynamic} other (non-UFP candidates, metric=MSE)."
            )
            if not self.ufp_candidates:
                print("  WARNING: no UFP formats in candidates; post-ReLU layers use non-UFP.")
            if not self.non_ufp_candidates:
                print("  WARNING: no non-UFP formats in candidates; other layers use UFP.")
        else:
            print(
                f"Registered hooks on {count_dynamic + count_ufp} layers: "
                f"all layers use all {len(self.candidate_formats)} candidates "
                f"(metric=MSE)."
            )
        if self.use_unsigned_input_candidates:
            print(f"{count_unsigned_candidate_layers} layers are using UFP candidates.")
        if self.skip_depthwise_input_quant:
            print(
                f"Depthwise input-quant ablation: skipped input quantization "
                f"for {count_skipped_depthwise} depthwise Conv2d layers."
            )

    def _formats_for_bits(self, bits, unsigned=False):
        def matches_width(fmt):
            if not isinstance(fmt, str) or not fmt.startswith(('fp', 'ufp')):
                return False
            width = fmt.split('_', 1)[0].lstrip('u')[2:]
            return width.isdigit() and int(width) == bits

        formats = [fmt for fmt in self.candidate_formats if matches_width(fmt)]
        if not formats:
            formats = [fmt for fmt in DEFAULT_DYNAMIC_INPUT_CANDIDATES if matches_width(fmt)]
        if unsigned:
            formats = self._make_unsigned_candidates(formats)
        return formats

    def _candidates_for_layer(self, layer_name, module=None, input_index=0):
        is_unsigned = self._layer_uses_unsigned_input(layer_name, input_index=input_index)

        if input_index == 1 and layer_name in self.layer_residual_input_bits_map:
            residual_bits = self.layer_residual_input_bits_map[layer_name]
            candidates = self._formats_for_bits(residual_bits, unsigned=is_unsigned)
            if candidates:
                return candidates

        stays_on_chip = self.cache_sim_map.get(layer_name)
        if stays_on_chip is None:
            if layer_name not in self._reported_missing_layers:
                print(f"[DynamicInputQuantizer] Layer '{layer_name}' NOT found in cache sim map. Using standard candidates.")
                self._reported_missing_layers.add(layer_name)
            
        if stays_on_chip:
            return self.unsigned_all_fp8_formats if is_unsigned else self.all_fp8_formats

        if is_unsigned:
            return self.unsigned_candidate_formats

        if self.restrict_post_relu_ufp:
            if layer_name in self.post_relu_layers:
                return self.ufp_candidates or self._make_unsigned_candidates(self.non_ufp_candidates)
            return self.non_ufp_candidates or self.ufp_candidates

        return self.candidate_formats

    def _should_quantize_input(self, layer_name, input_index=0):
        # On-chip-resident inputs are NOT exempt from quantization: the simulated
        # architecture stores activations at the on-chip width (8-bit) inside the
        # chip, so they must still be quantized. The need_input_transfer map only
        # affects the cycle/runtime accounting in compute_model_runtime, not
        # whether the value is quantized here. Always quantize the primary input;
        # _candidates_for_layer already supplies 8-bit formats for on-chip layers.
        return True

    def _get_hook(self, layer_name):
        def hook_fn(module, args):
            x = args[0]
            if not isinstance(x, torch.Tensor):
                return None

            quantize_primary = self._should_quantize_input(layer_name, input_index=0)
            if quantize_primary:
                candidates = self._candidates_for_layer(layer_name, module, input_index=0)
                x_quantized, best_indices = self._quantize_input_tensor(x, layer_name, candidates, module)
                self._set_module_input_quant_state(module, candidates, best_indices)
            else:
                # The input is already resident on-chip for this layer, so do not
                # model an external-memory transfer quantization for it.
                module.input_quantization = False
                module.input_mode = 'fp32'
                x_quantized = x

            if (
                module.__class__.__name__ == "QuantAdd"
                and len(args) >= 2
                and isinstance(args[1], torch.Tensor)
                and layer_name in self.layer_residual_input_bits_map
                and self._should_quantize_input(layer_name, input_index=1)
            ):
                residual_candidates = self._candidates_for_layer(layer_name, module, input_index=1)
                residual_quantized, residual_indices = self._quantize_input_tensor(
                    args[1], f"{layer_name}.input2", residual_candidates, module
                )
                self._set_module_input_quant_state(
                    module, residual_candidates, residual_indices, input_index=1
                )
                return (x_quantized, residual_quantized, *args[2:])

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
                query, input_layer_name("query_input"), candidates, module
            )
            k_quantized, _ = self._quantize_input_tensor(
                key, input_layer_name("key_input"), candidates, module
            )
            v_quantized, _ = self._quantize_input_tensor(
                value, input_layer_name("value_input"), candidates, module
            )
            self._set_module_input_quant_state(module, candidates, q_indices)

            return (q_quantized, k_quantized, v_quantized, *args[3:])

        return hook_fn

    def _set_module_input_quant_state(self, module, candidates, best_indices, input_index=0):
        # Disable a quantized module's built-in input quantization path after the
        # pre-hook has already supplied dynamically quantized inputs.
        module.input_quantization = False
        module.input_mode = 'chunk'
        module.input_chunk_size = self.chunk_size

        # Store candidates and indices without turning the full tensor into a
        # Python list; that would force a large GPU-CPU sync.
        if input_index == 0:
            module.input_chunk_candidates = candidates
            module.input_chunk_format_indices = best_indices
        else:
            prefix = f'input{input_index + 1}'
            setattr(module, f'{prefix}_chunk_candidates', candidates)
            setattr(module, f'{prefix}_chunk_format_indices', best_indices)

        if len(candidates) > 0:
            if input_index > 0:
                setattr(module, f'input{input_index + 1}_q_type', candidates[0])
            else:
                module.input_q_type = candidates[0]
                if self._is_multi_input_module(module):
                    setattr(module, 'input1_q_type', candidates[0])

        module.rounding = 'nearest'

    def _quantize_input_tensor(self, tensor, layer_name, candidates, module=None):
        quantized, best_indices = self._select_best_format(tensor, layer_name, candidates, module)

        if self.collect_error_stats:
            with torch.no_grad():
                diff = tensor - quantized
                updates = {
                    'sum_mse_err': diff.pow(2).sum(),
                    'sum_l2_norm': tensor.pow(2).sum(),
                }
                for key, value in updates.items():
                    if self.stats[key] is None:
                        self.stats[key] = value.detach()
                    else:
                        self.stats[key] += value.detach()

        return quantized, best_indices

    def _candidate_params(self, candidates):
        key = tuple(candidates)
        cached = self._candidate_param_cache.get(key)
        if cached is not None:
            return cached

        cands_e = []
        cands_m = []
        cands_sgn = []
        for fmt in candidates:
            # The CUDA kernel does not natively bypass fp32. Map it to a
            # high-precision FP format for compatibility with older configs.
            if fmt == 'fp32':
                cands_e.append(8)
                cands_m.append(7)
                cands_sgn.append(1)
            else:
                e, m = get_format_params(fmt)
                is_signed = not fmt.startswith('ufp')
                cands_e.append(e)
                cands_m.append(m)
                cands_sgn.append(1 if is_signed else 0)

        cached = (cands_e, cands_m, cands_sgn)
        self._candidate_param_cache[key] = cached
        return cached

    def _select_best_format(self, tensor, layer_name, candidates, module=None):
        """
        Optimized format selection:
        - Iterates through candidates and keeps only the best per-chunk to save VRAM.
        - Returns the quantized tensor and the TENSOR of best indices.
        """
        chunk_size = self.chunk_size
        device = tensor.device
        capture = getattr(module, 'capture_activations', False)

        ref_chunked, original_shape, pad_len = chunk_tensor_by_context(
            tensor, chunk_size
        )
        num_contexts, num_chunks, chunk_size = ref_chunked.shape
        ref_chunks = ref_chunked.reshape(-1, chunk_size)
        total_chunks = ref_chunks.shape[0]

        if len(candidates) == 1:
            best_indices = torch.zeros(total_chunks, dtype=torch.long, device=device)
            if capture:
                best_qs, best_unscaled_qs, _max_val, scale_chunks, _scale_p = quantize_tensor(
                    ref_chunks.contiguous(),
                    q_type=candidates[0],
                    return_unscaled=True,
                    return_scale=True,
                    mode='chunk',
                    chunk_size=chunk_size,
                )
            else:
                best_qs, _max_val = quantize_tensor(
                    ref_chunks.contiguous(),
                    q_type=candidates[0],
                    mode='chunk',
                    chunk_size=chunk_size,
                )
                scale_chunks = None
        else:
            cands_e, cands_m, cands_sgn = self._candidate_params(candidates)

            # Call the fused CUDA kernel
            best_indices, best_scales, best_qs_flat, best_unscaled_qs_flat = search_best_chunk_format(
                ref_chunks.view(-1).contiguous(),
                cands_e,
                cands_m,
                cands_sgn,
                capture
            )

            best_qs = best_qs_flat.view(-1, chunk_size)
            if capture:
                best_unscaled_qs = best_unscaled_qs_flat.view(-1, chunk_size)
                scale_chunks = best_scales.unsqueeze(-1).expand(-1, chunk_size).contiguous()
            else:
                scale_chunks = None

        # Reconstruct the quantized tensor from best chunks
        q_reshaped = best_qs.view(num_contexts, num_chunks, chunk_size)
        quantized_tensor = unchunk_tensor_by_context(q_reshaped, original_shape, pad_len)

        # Populate activation capture fields if enabled
        if capture:
            u_reshaped = best_unscaled_qs.view(num_contexts, num_chunks, chunk_size)
            module.last_quant_input_unscaled = unchunk_tensor_by_context(u_reshaped, original_shape, pad_len).detach()
            module.last_quant_input = quantized_tensor.detach()
            s_reshaped = scale_chunks.view(num_contexts, num_chunks, chunk_size)
            module.last_quant_input_scale = unchunk_tensor_by_context(s_reshaped, original_shape, pad_len).detach()

        if self.collect_format_stats:
            # Update layer-wise format selection statistics (stay on GPU).
            if layer_name not in self.layer_stats:
                self.layer_stats[layer_name] = {
                    'format_counts_tensor': torch.zeros(len(candidates), dtype=torch.long, device=device),
                    'candidates': candidates
                }
            
            stats = self.layer_stats[layer_name]
            counts = torch.bincount(best_indices, minlength=len(candidates))
            stats['format_counts_tensor'] += counts

        # Track running error for progress/debug
        if self.running_error is None:
            self.running_error = 0.0
        # self.running_error += best_errors.sum().item() # Removed for CUDA optimization
        self.total_chunks += total_chunks

        return quantized_tensor, best_indices

    def get_final_stats(self):
        scalar_stats = {
            key: (value.item() if isinstance(value, torch.Tensor) else 0.0)
            for key, value in self.stats.items()
        }
        norm_mse = (
            scalar_stats['sum_mse_err'] / scalar_stats['sum_l2_norm']
            if scalar_stats['sum_l2_norm'] > 0 else 0.0
        )
        
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
                    'total_chunks': int(counts_tensor.sum().item()),
                    'stays_on_chip': self.cache_sim_map.get(layer_name, True)
                }
            else:
                stats_copy = dict(stats)
                stats_copy['stays_on_chip'] = self.cache_sim_map.get(layer_name, True)
                processed_layer_stats[layer_name] = stats_copy

        return {
            'norm_mse': norm_mse,
            'total_mse': scalar_stats['sum_mse_err'],
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

        for module in self.skipped_depthwise_modules:
            module.input_quantization = True
            if hasattr(module, 'input_mode'):
                module.input_mode = 'chunk'
        self.skipped_depthwise_modules = []

        self.hooked_modules = []
