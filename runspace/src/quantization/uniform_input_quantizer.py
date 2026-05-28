import torch
import torch.nn as nn

try:
    from ..registry.op_registry import OpRegistry
    from ..ops.quant_base import quantize_tensor
    from .chunking import count_context_chunks
except ImportError:
    from src.registry.op_registry import OpRegistry
    from src.ops.quant_base import quantize_tensor
    from src.quantization.chunking import count_context_chunks


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

    def __init__(
        self,
        model,
        fmt,
        chunk_size=128,
        quant_mode='chunk',
        unsigned_input_sources=None,
        use_unsigned_input_candidates=True,
    ):
        self.model = model
        self.fmt = fmt
        self.unsigned_fmt = self._to_unsigned_format(fmt)
        self.chunk_size = chunk_size
        self.quant_mode = quant_mode
        self.use_unsigned_input_candidates = bool(use_unsigned_input_candidates)
        self.unsigned_input_sources = {
            str(source).lower()
            for source in (unsigned_input_sources or [])
            if str(source).strip()
        }
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
        self.unsigned_passthrough_layers = set()
        self.post_unsigned_layers = self._find_post_unsigned_layers()
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
            mode=self.quant_mode,
            chunk_size=self.chunk_size if self.quant_mode == 'chunk' else None,
        )
        return x_q

    @staticmethod
    def _to_unsigned_format(fmt):
        if not isinstance(fmt, str) or fmt == 'fp32' or fmt.startswith(('ufp', 'uefp')):
            return fmt
        if not fmt.startswith(('fp', 'efp')):
            return fmt

        try:
            from src.ops.quant_softmax import qtype_to_unsigned_qtype
        except ImportError:
            from ..ops.quant_softmax import qtype_to_unsigned_qtype

        return qtype_to_unsigned_qtype(fmt, add_to_mant=True)

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
    def _is_multi_input_module(module):
        class_name = module.__class__.__name__.lower()
        unquantized_name = class_name.replace('quant', '')
        return unquantized_name in ('add', 'mul', 'sub', 'div', 'matmul', 'bmm', 'cat')

    @staticmethod
    def _is_passthrough_module(module):
        return isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.Identity))

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

                is_compute = isinstance(module, self.hookable_ops) or isinstance(module, (nn.Conv2d, nn.Linear))
                if not is_compute:
                    continue

                node_inputs = self._node_inputs(node)
                if len(node_inputs) > 1:
                    is_post_unsigned = node_inputs[0].name in unsigned_nodes
                else:
                    is_post_unsigned = any(inp.name in unsigned_nodes for inp in node_inputs)

                if self._module_uses_unsigned_input(module) or is_post_unsigned:
                    layer_name = f"{prefix}.{node.target}" if prefix else str(node.target)
                    post_unsigned.add(layer_name)

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
            is_compute = isinstance(module, self.hookable_ops) or isinstance(module, (nn.Conv2d, nn.Linear))
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
                if prev_was_unsigned and not self._is_multi_input_module(module):
                    post_unsigned.add(name)
                elif uses_unsigned_input:
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

    def _effective_format_for_module(self, layer_name, module):
        if (
            self.use_unsigned_input_candidates
            and (
                layer_name in self.post_unsigned_layers
                or layer_name in self.unsigned_passthrough_layers
                or self._module_uses_unsigned_input(module)
            )
        ):
            return self.unsigned_fmt

        return self.fmt

    def _quantize_with_format(self, x, fmt):
        x_q, _ = quantize_tensor(
            x,
            q_type=fmt,
            mode=self.quant_mode,
            chunk_size=self.chunk_size if self.quant_mode == 'chunk' else None,
        )
        return x_q

    def _make_hook(self, layer_name):
        def hook(module, inputs):
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return None

            x = inputs[0]
            fmt = self._effective_format_for_module(layer_name, module)
            x_q = self._quantize_with_format(x, fmt)

            if self.quant_mode == 'chunk':
                total_chunks = count_context_chunks(x, self.chunk_size)
            else:
                total_chunks = x.shape[0] if x.dim() > 0 else 1

            module.input_quantization = True
            module.input_mode = self.quant_mode
            module.input_chunk_size = self.chunk_size if self.quant_mode == 'chunk' else None
            module.input_q_type = fmt
            if self._is_multi_input_module(module):
                module.input1_q_type = fmt
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
            stats['format_counts'][fmt] = stats['format_counts'].get(fmt, 0) + total_chunks
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
            'layer_stats': self.layer_stats,
        }
