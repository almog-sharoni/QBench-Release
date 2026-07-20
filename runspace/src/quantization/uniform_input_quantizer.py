from collections import deque

import torch
import torch.nn as nn

try:
    from ..registry.op_registry import OpRegistry
    from ..ops.quant_base import quantize_tensor
    from .chunking import count_context_chunks, chunk_tensor_by_context, unchunk_tensor_by_context
    from .activation_transport import (
        ActivationPacket,
        ActivationTransport,
        normalize_activation_transport,
    )
    from .activation_transport_runtime import ActivationBypass, ActivationTransportRuntime
except ImportError:
    from src.registry.op_registry import OpRegistry
    from src.ops.quant_base import quantize_tensor
    from src.quantization.chunking import count_context_chunks, chunk_tensor_by_context, unchunk_tensor_by_context
    from src.quantization.activation_transport import (
        ActivationPacket,
        ActivationTransport,
        normalize_activation_transport,
    )
    from src.quantization.activation_transport_runtime import (
        ActivationBypass,
        ActivationTransportRuntime,
    )


class UniformInputQuantizer:
    """
    Applies one fixed format at producer-stage activation boundaries.

    Encoded transport is the hardware path. Reference transport keeps the same
    FX boundary plan while carrying decoded FP32 tensors for development.
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
        collect_error_stats=True,
        transport="encoded",
        stage_format_policy=None,
    ):
        self.model = model
        self.fmt = fmt
        self.unsigned_fmt = self._to_unsigned_format(fmt)
        self.chunk_size = chunk_size
        self.quant_mode = quant_mode
        self.use_unsigned_input_candidates = bool(use_unsigned_input_candidates)
        self.collect_error_stats = bool(collect_error_stats)
        self.transport = normalize_activation_transport(transport)
        self.activation_transport = None
        self._transport_runtime = None
        self.stage_format_policy = self._normalize_stage_format_policy(
            stage_format_policy
        )
        self._stage_formats = {}
        self._graph_nodes = {}
        self._pending_error_inputs = {}
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
        self.layer_unsigned_input_indices = {}
        self.post_unsigned_layers = self._find_post_unsigned_layers()
        self.stats = {
            'sum_l1_err': None,
            'sum_mse_err': None,
            'sum_l1_norm': None,
            'sum_l2_norm': None,
        }

    @staticmethod
    def _normalize_stage_format_policy(policy):
        if policy is None:
            return None
        if not isinstance(policy, dict):
            raise TypeError("stage_format_policy must be a mapping")

        normalized = {}
        for key in ('producer_default', 'consumer_default'):
            value = policy.get(key)
            normalized[key] = str(value) if value is not None else None
        for key in ('producer_overrides', 'consumer_overrides'):
            value = policy.get(key, {})
            if not isinstance(value, dict):
                raise TypeError(f"stage_format_policy.{key} must be a mapping")
            normalized[key] = {
                str(layer_name): str(fmt)
                for layer_name, fmt in value.items()
            }
        return normalized

    def _stage_format(self, stage):
        output_node = self._graph_nodes.get(stage.output_node)
        if output_node is not None and output_node.meta.get(
            "qbench_activation_bypass"
        ):
            return None
        if self.stage_format_policy is None:
            selected = self.fmt
        else:
            producer_default = self.stage_format_policy['producer_default']
            producer_overrides = self.stage_format_policy['producer_overrides']
            producer_targets = []
            for node_name in stage.node_names:
                node = self._graph_nodes.get(node_name)
                if node is None or node.op != 'call_module':
                    continue
                producer_targets.append(str(node.target))

            output_target = (
                str(output_node.target)
                if output_node is not None and output_node.op == 'call_module'
                else None
            )
            if output_target in producer_overrides:
                producer_requests = [
                    (f"producer {output_target}", producer_overrides[output_target])
                ]
            else:
                producer_requests = [
                    (f"producer {target}", producer_overrides[target])
                    for target in producer_targets
                    if target in producer_overrides
                ]
                if not producer_requests and producer_targets and producer_default is not None:
                    producer_requests = [
                        (f"producer {producer_targets[-1]}", producer_default)
                    ]

            producer_formats = {
                fmt.strip().lower()
                for _source, fmt in producer_requests
                if fmt.strip().lower() != 'fp32'
            }
            if len(producer_formats) > 1:
                details = ', '.join(
                    f"{source}={fmt}" for source, fmt in producer_requests
                )
                raise ValueError(
                    f"Producer stage {stage.stage_id!r} has incompatible legacy "
                    f"output formats ({details}); one hardware packet must be "
                    "shared by the fused producer stage."
                )
            producer_format = next(iter(producer_formats), None)

            consumer_default = self.stage_format_policy['consumer_default']
            consumer_overrides = self.stage_format_policy['consumer_overrides']
            consumer_requests = []
            for node_name in stage.consumer_nodes:
                node = self._graph_nodes.get(node_name)
                if node is None or node.op != 'call_module':
                    continue
                target = str(node.target)
                fmt = consumer_overrides.get(target, consumer_default)
                if fmt is not None:
                    consumer_requests.append((f"consumer {target}", fmt))

            consumer_formats = {
                fmt.strip().lower() for _source, fmt in consumer_requests
            }
            if producer_format is not None:
                incompatible = {
                    fmt for fmt in consumer_formats
                    if fmt != 'fp32' and fmt != producer_format
                }
                selected = producer_format
            else:
                incompatible = consumer_formats if len(consumer_formats) > 1 else set()
                selected = next(iter(consumer_formats), None)

            if incompatible:
                requests = producer_requests + consumer_requests
                details = ', '.join(
                    f"{source}={fmt}" for source, fmt in requests
                )
                raise ValueError(
                    f"Producer stage {stage.stage_id!r} has incompatible legacy "
                    f"activation formats ({details}); one hardware packet must be "
                    "shared by every consumer. Configure evaluation.input_quant "
                    "with one producer-stage policy."
                )

        if selected is None or selected.strip().lower() == 'fp32':
            return None
        if stage.is_unsigned and self.use_unsigned_input_candidates:
            return self._to_unsigned_format(selected)
        return selected

    def _quantize_context(self, x, fmt):
        """Per-context fixed-format quant via chunk_tensor_by_context.

        Mirrors the dynamic input quantizer's chunking (one scale per per-context
        128-chunk) so fixed-format input baselines are directly comparable to the
        dynamic per-chunk runs. quantize_tensor(mode='chunk') is used only as the
        codec PE primitive on the already-formed [N,128] rows.
        """
        chunked, original_shape, pad_len = chunk_tensor_by_context(x, self.chunk_size)
        b, nc, cs = chunked.shape
        deq, _ = quantize_tensor(chunked.reshape(-1, cs), q_type=fmt, mode='chunk', chunk_size=cs)
        return unchunk_tensor_by_context(deq.view(b, nc, cs), original_shape, pad_len)

    def _quantize(self, x):
        if self.quant_mode == 'chunk':
            return self._quantize_context(x, self.fmt)
        x_q, _ = quantize_tensor(x, q_type=self.fmt, mode=self.quant_mode, chunk_size=None)
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

    def _mark_unsigned_input(self, layer_name, input_index=0):
        self.layer_unsigned_input_indices.setdefault(layer_name, set()).add(int(input_index))

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
                    unsigned_input_indices = {
                        idx for idx, inp in enumerate(node_inputs)
                        if inp.name in unsigned_nodes
                    }
                    is_post_unsigned = bool(unsigned_input_indices)
                else:
                    unsigned_input_indices = {0} if any(inp.name in unsigned_nodes for inp in node_inputs) else set()
                    is_post_unsigned = bool(unsigned_input_indices)

                if self._module_uses_unsigned_input(module) or is_post_unsigned:
                    layer_name = f"{prefix}.{node.target}" if prefix else str(node.target)
                    post_unsigned.add(layer_name)
                    for input_index in unsigned_input_indices:
                        self._mark_unsigned_input(layer_name, input_index)

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
                    self._mark_unsigned_input(name, 0)
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

    def _effective_input_formats_for_module(self, layer_name, module, input_count):
        formats = [self.fmt for _ in range(input_count)]
        if not self.use_unsigned_input_candidates:
            return formats

        unsigned_indices = set(self.layer_unsigned_input_indices.get(layer_name, set()))
        if not unsigned_indices and (
            layer_name in self.post_unsigned_layers
            or layer_name in self.unsigned_passthrough_layers
            or self._module_uses_unsigned_input(module)
        ):
            unsigned_indices.add(0)

        for idx in unsigned_indices:
            if 0 <= idx < input_count:
                formats[idx] = self.unsigned_fmt
        return formats

    def _quantize_with_format(self, x, fmt):
        if self.quant_mode == 'chunk':
            return self._quantize_context(x, fmt)
        x_q, _ = quantize_tensor(x, q_type=fmt, mode=self.quant_mode, chunk_size=None)
        return x_q

    def _make_hook(self, layer_name):
        def hook(module, inputs):
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return None

            x = inputs[0]
            fmt = self._effective_format_for_module(layer_name, module)
            input_formats = None
            if self._is_multi_input_module(module):
                tensor_inputs = [arg for arg in inputs if isinstance(arg, torch.Tensor)]
                input_formats = self._effective_input_formats_for_module(
                    layer_name,
                    module,
                    len(tensor_inputs),
                )
                if input_formats:
                    fmt = input_formats[0]
            x_q = self._quantize_with_format(x, fmt)

            if self.quant_mode == 'chunk':
                total_chunks = count_context_chunks(x, self.chunk_size)
            else:
                total_chunks = x.shape[0] if x.dim() > 0 else 1

            module.input_quantization = True
            module.input_mode = self.quant_mode
            module.input_chunk_size = self.chunk_size if self.quant_mode == 'chunk' else None
            if self._is_multi_input_module(module):
                module.input_q_type = self.fmt
                for idx, input_fmt in enumerate(input_formats or [fmt], start=1):
                    setattr(module, f'input{idx}_q_type', input_fmt)
            else:
                module.input_q_type = fmt
            module.input_chunk_formats = None
            module.rounding = 'nearest'

            if self.collect_error_stats:
                with torch.no_grad():
                    diff = x - x_q
                    updates = {
                        'sum_l1_err': diff.abs().sum(),
                        'sum_mse_err': diff.pow(2).sum(),
                        'sum_l1_norm': x.abs().sum(),
                        'sum_l2_norm': x.pow(2).sum(),
                    }
                    for key, value in updates.items():
                        value = value.detach().to(dtype=torch.float64)
                        if self.stats[key] is None:
                            self.stats[key] = value
                        else:
                            self.stats[key].add_(value)

            stats = self.layer_stats.setdefault(
                layer_name,
                {'format_counts': {}, 'total_chunks': 0, 'type': module.__class__.__name__}
            )
            stats['format_counts'][fmt] = stats['format_counts'].get(fmt, 0) + total_chunks
            stats['total_chunks'] += total_chunks

            return None

        return hook

    def register_hooks(self):
        """Install producer-stage hardware transport as the only runtime path."""
        if self._transport_runtime is not None:
            return
        if self.transport == "encoded" and self.quant_mode != "chunk":
            raise ValueError(
                "Encoded activation transport only supports quant_mode='chunk'; "
                f"got {self.quant_mode!r}."
            )

        self.activation_transport = ActivationTransport(
            mode=self.transport,
            chunk_size=int(self.chunk_size),
        )

        def encode_stage(stage, tensor):
            fmt = self._stage_formats[stage.stage_id]
            if fmt is None:
                return ActivationBypass(tensor)
            if self.quant_mode == "chunk":
                transmitted = self.activation_transport.transmit_uniform(
                    tensor,
                    fmt,
                    producer_id=stage.stage_id,
                )
            else:
                transmitted = self._quantize_with_format(tensor, fmt)

            if self.collect_error_stats:
                self._pending_error_inputs.setdefault(
                    stage.stage_id,
                    deque(),
                ).append(tensor.detach())

            if isinstance(transmitted, ActivationPacket):
                total_chunks = transmitted.num_chunks
            elif self.quant_mode == 'chunk':
                total_chunks = count_context_chunks(tensor, self.chunk_size)
            else:
                total_chunks = tensor.shape[0] if tensor.dim() > 0 else 1
            stats = self.layer_stats.setdefault(
                stage.stage_id,
                {
                    'layer_name': self._transport_runtime.stage_display_name(stage),
                    'stage_id': stage.stage_id,
                    'format_counts': {},
                    'total_chunks': 0,
                    'type': stage.kind.value,
                    'producer_nodes': list(stage.node_names),
                    'consumer_nodes': list(stage.consumer_nodes),
                    'is_unsigned': bool(stage.is_unsigned),
                    'unsigned_source': stage.unsigned_source,
                    'candidate_formats': [fmt],
                },
            )
            stats['format_counts'][fmt] = stats['format_counts'].get(fmt, 0) + total_chunks
            stats['total_chunks'] += total_chunks
            return transmitted

        def observe_decode(stage_id, quantized):
            pending = self._pending_error_inputs.get(stage_id)
            if not pending:
                return
            tensor = pending.popleft()
            with torch.no_grad():
                diff = tensor - quantized
                updates = {
                    'sum_l1_err': diff.abs().sum(),
                    'sum_mse_err': diff.pow(2).sum(),
                    'sum_l1_norm': tensor.abs().sum(),
                    'sum_l2_norm': tensor.pow(2).sum(),
                }
                for key, value in updates.items():
                    value = value.detach().to(dtype=torch.float64)
                    if self.stats[key] is None:
                        self.stats[key] = value
                    else:
                        self.stats[key].add_(value)

        self._transport_runtime = ActivationTransportRuntime(
            self.model,
            self.activation_transport,
            encode_stage,
            decode_observer=observe_decode,
        ).install()
        plan = self._transport_runtime.plan
        self._graph_nodes = {node.name: node for node in plan.graph_module.graph.nodes}
        try:
            self._stage_formats = {
                stage.stage_id: self._stage_format(stage)
                for stage in plan.stages
            }
        except Exception:
            self._transport_runtime.cleanup()
            self._transport_runtime = None
            self._graph_nodes = {}
            raise
        unsigned_stages = sum(stage.is_unsigned for stage in plan.stages)
        quantized_stages = sum(fmt is not None for fmt in self._stage_formats.values())
        print(
            "Activation transport enabled: "
            f"mode=uniform transport={self.transport} format={self.fmt} "
            f"chunk_size={self.chunk_size} stages={quantized_stages}/{len(plan.stages)} "
            f"unsigned_stages={unsigned_stages}"
        )

    def cleanup(self):
        if self._transport_runtime is not None:
            self._transport_runtime.cleanup()
            self._transport_runtime = None
        self._stage_formats = {}
        self._graph_nodes = {}
        self._pending_error_inputs.clear()
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_final_stats(self):
        scalar_stats = {
            key: (float(value.item()) if isinstance(value, torch.Tensor) else 0.0)
            for key, value in self.stats.items()
        }
        norm_l1 = (
            scalar_stats['sum_l1_err'] / scalar_stats['sum_l1_norm']
            if scalar_stats['sum_l1_norm'] > 0 else 0.0
        )
        norm_mse = (
            scalar_stats['sum_mse_err'] / scalar_stats['sum_l2_norm']
            if scalar_stats['sum_l2_norm'] > 0 else 0.0
        )
        result = {
            'norm_l1': norm_l1,
            'norm_mse': norm_mse,
            'total_l1': scalar_stats['sum_l1_err'],
            'total_mse': scalar_stats['sum_mse_err'],
            'layer_stats': self.layer_stats,
            'collect_error_stats': self.collect_error_stats,
        }
        if self._transport_runtime is not None:
            result.update(self._transport_runtime.transport_stats())
        else:
            result.update(
                {
                    'transport': self.transport,
                    'transmission_count': 0,
                    'packet_count': 0,
                    'decode_reads': 0,
                    'encoded_bytes': 0,
                    'planner_version': 1,
                    'stage_count': 0,
                    'unsigned_stage_count': 0,
                    'activation_plan': {},
                }
            )
        return result
