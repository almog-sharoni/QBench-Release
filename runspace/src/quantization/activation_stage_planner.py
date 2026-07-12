"""Strict FX planning for hardware activation transport boundaries.

The planner is intentionally separate from packet encoding and runtime graph
rewriting.  It describes where a tensor must be encoded, which nodes may carry
the encoded packet without requantization, and where the packet is consumed.
Unknown value-producing operations are rejected so an unsupported model cannot
silently fall back to an incorrect boundary.
"""

from __future__ import annotations

import builtins
import operator
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable, Iterable, Mapping

import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import stochastic_depth


class NodeRole(str, Enum):
    """A node's effect on activation transport."""

    INPUT = "input"
    CONSTANT = "constant"
    COMPUTE = "compute"
    ACTIVATION = "activation"
    TRANSPARENT = "transparent"
    NON_TENSOR = "non_tensor"
    OUTPUT = "output"


class StageKind(str, Enum):
    """The operation that starts an activation stage."""

    INPUT = "input"
    COMPUTE = "compute"
    ACTIVATION = "activation"


@dataclass(frozen=True)
class NodePolicy:
    """Classification used by the planner for one FX operation."""

    role: NodeRole
    is_unsigned: bool = False
    unsigned_source: str | None = None

    def __post_init__(self) -> None:
        if self.is_unsigned and self.role not in {
            NodeRole.INPUT,
            NodeRole.COMPUTE,
            NodeRole.ACTIVATION,
        }:
            raise ValueError(f"{self.role.value} nodes cannot produce unsigned packets")
        if self.unsigned_source is not None and not self.is_unsigned:
            raise ValueError("unsigned_source requires is_unsigned=True")


@dataclass(frozen=True)
class ActivationStage:
    """One producer and its optional, exclusively fused activation."""

    stage_id: str
    kind: StageKind
    node_names: tuple[str, ...]
    entry_node: str
    output_node: str
    input_stage_ids: tuple[str, ...]
    consumer_nodes: tuple[str, ...] = ()
    consumer_stage_ids: tuple[str, ...] = ()
    passthrough_nodes: tuple[str, ...] = ()
    is_unsigned: bool = False
    unsigned_source: str | None = None
    has_fanout: bool = False


@dataclass(frozen=True)
class ActivationStagePlan:
    """Complete activation-boundary plan for one FX graph."""

    graph_module: fx.GraphModule
    stages: tuple[ActivationStage, ...]
    node_to_stage: Mapping[str, str]
    node_roles: Mapping[str, NodeRole]
    node_sources: Mapping[str, tuple[str, ...]]
    model_output_sources: tuple[str, ...]

    def stage(self, stage_id: str) -> ActivationStage:
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        raise KeyError(f"Unknown activation stage {stage_id!r}")

    def stage_for_node(self, node_name: str) -> ActivationStage:
        try:
            stage_id = self.node_to_stage[node_name]
        except KeyError as exc:
            raise KeyError(f"FX node {node_name!r} is not owned by a stage") from exc
        return self.stage(stage_id)

    @property
    def boundary_nodes(self) -> tuple[str, ...]:
        return tuple(stage.output_node for stage in self.stages)


class ActivationStagePlanningError(RuntimeError):
    """Base error for graphs that cannot be planned without guessing."""


class UnsupportedNodeError(ActivationStagePlanningError):
    """Raised for a value-producing node without an explicit policy."""


PolicyLike = NodePolicy | NodeRole | str


_ACTIVATION_MODULE_POLICIES: tuple[tuple[type[nn.Module], NodePolicy], ...] = (
    (nn.ReLU, NodePolicy(NodeRole.ACTIVATION, True, "relu")),
    (nn.ReLU6, NodePolicy(NodeRole.ACTIVATION, True, "relu6")),
    (nn.Softmax, NodePolicy(NodeRole.ACTIVATION, True, "softmax")),
    (nn.Sigmoid, NodePolicy(NodeRole.ACTIVATION, True, "sigmoid")),
    (nn.GELU, NodePolicy(NodeRole.ACTIVATION)),
    (nn.SiLU, NodePolicy(NodeRole.ACTIVATION)),
    (nn.Hardswish, NodePolicy(NodeRole.ACTIVATION)),
    (nn.Hardsigmoid, NodePolicy(NodeRole.ACTIVATION, True, "hardsigmoid")),
)

_ACTIVATION_NAME_POLICIES: Mapping[str, NodePolicy] = {
    "quantrelu": NodePolicy(NodeRole.ACTIVATION, True, "relu"),
    "quantrelu6": NodePolicy(NodeRole.ACTIVATION, True, "relu6"),
    "quantsoftmax": NodePolicy(NodeRole.ACTIVATION, True, "softmax"),
    "quantgelu": NodePolicy(NodeRole.ACTIVATION),
    "quantsilu": NodePolicy(NodeRole.ACTIVATION),
    "quanthardswish": NodePolicy(NodeRole.ACTIVATION),
    "quanthardsigmoid": NodePolicy(NodeRole.ACTIVATION, True, "hardsigmoid"),
}

_DROPOUT_MODULE_TYPES = (
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
    nn.FeatureAlphaDropout,
)

_TRANSPARENT_MODULE_TYPES = (
    nn.Identity,
    *_DROPOUT_MODULE_TYPES,
    nn.Flatten,
    nn.Unflatten,
)

_TRANSPARENT_MODULE_NAMES = {
    "quantidentity",
    "quantdropout",
    "quantdropout1d",
    "quantdropout2d",
    "quantdropout3d",
}

_COMPUTE_MODULE_TYPES = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.Linear,
    nn.Bilinear,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.LayerNorm,
    nn.GroupNorm,
    nn.LocalResponseNorm,
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.MaxPool3d,
    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.AvgPool3d,
    nn.AdaptiveAvgPool1d,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveAvgPool3d,
    nn.AdaptiveMaxPool1d,
    nn.AdaptiveMaxPool2d,
    nn.AdaptiveMaxPool3d,
    nn.Embedding,
    nn.EmbeddingBag,
)

_QUANT_COMPUTE_NAMES = {
    "quantconv1d",
    "quantconv2d",
    "quantconv3d",
    "quantconvtranspose1d",
    "quantconvtranspose2d",
    "quantconvtranspose3d",
    "quantlinear",
    "quantbatchnorm1d",
    "quantbatchnorm2d",
    "quantbatchnorm3d",
    "quantlayernorm",
    "quantgroupnorm",
    "quantmaxpool1d",
    "quantmaxpool2d",
    "quantmaxpool3d",
    "quantavgpool1d",
    "quantavgpool2d",
    "quantavgpool3d",
    "quantadaptiveavgpool1d",
    "quantadaptiveavgpool2d",
    "quantadaptiveavgpool3d",
    "quantmatmul",
    "quantbmm",
    "quantadd",
    "quantsub",
    "quantmul",
    "quantdiv",
    "quantcat",
}

_ACTIVATION_FUNCTION_POLICIES: Mapping[Callable[..., Any], NodePolicy] = {
    F.relu: NodePolicy(NodeRole.ACTIVATION, True, "relu"),
    torch.relu: NodePolicy(NodeRole.ACTIVATION, True, "relu"),
    F.relu6: NodePolicy(NodeRole.ACTIVATION, True, "relu6"),
    F.softmax: NodePolicy(NodeRole.ACTIVATION, True, "softmax"),
    torch.softmax: NodePolicy(NodeRole.ACTIVATION, True, "softmax"),
    F.sigmoid: NodePolicy(NodeRole.ACTIVATION, True, "sigmoid"),
    torch.sigmoid: NodePolicy(NodeRole.ACTIVATION, True, "sigmoid"),
    F.gelu: NodePolicy(NodeRole.ACTIVATION),
    F.silu: NodePolicy(NodeRole.ACTIVATION),
    F.hardswish: NodePolicy(NodeRole.ACTIVATION),
    F.hardsigmoid: NodePolicy(NodeRole.ACTIVATION, True, "hardsigmoid"),
}

_TRANSPARENT_FUNCTIONS = {
    operator.getitem,
    torch.flatten,
    torch.reshape,
    torch.squeeze,
    torch.unsqueeze,
    torch.transpose,
}
if hasattr(torch, "permute"):
    _TRANSPARENT_FUNCTIONS.add(torch.permute)

_DROPOUT_FUNCTIONS = {
    F.dropout,
    F.dropout1d,
    F.dropout2d,
    F.dropout3d,
    F.alpha_dropout,
    F.feature_alpha_dropout,
}

_STOCHASTIC_DEPTH_FUNCTIONS = {stochastic_depth}

_COMPUTE_FUNCTIONS = {
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.mod,
    operator.pow,
    operator.matmul,
    torch.add,
    torch.sub,
    torch.mul,
    torch.div,
    torch.matmul,
    torch.mm,
    torch.bmm,
    torch.einsum,
    torch.cat,
    torch.stack,
    torch.where,
    torch.maximum,
    torch.minimum,
    torch.abs,
    torch.neg,
    torch.exp,
    torch.exp2,
    torch.log,
    torch.log2,
    torch.sqrt,
    torch.rsqrt,
    torch.sigmoid,
    torch.tanh,
    torch.floor,
    torch.ceil,
    torch.clamp,
    torch.sum,
    torch.mean,
    torch.amax,
    torch.amin,
    F.linear,
    F.conv1d,
    F.conv2d,
    F.conv3d,
    F.conv_transpose1d,
    F.conv_transpose2d,
    F.conv_transpose3d,
    F.batch_norm,
    F.layer_norm,
    F.group_norm,
    F.max_pool1d,
    F.max_pool2d,
    F.max_pool3d,
    F.avg_pool1d,
    F.avg_pool2d,
    F.avg_pool3d,
    F.adaptive_avg_pool1d,
    F.adaptive_avg_pool2d,
    F.adaptive_avg_pool3d,
    F.interpolate,
    F.pad,
}

_NON_TENSOR_METHODS = {
    "size",
    "dim",
    "ndimension",
    "numel",
    "stride",
    "item",
    "tolist",
    "__len__",
}

_TRANSPARENT_METHODS = {
    "view",
    "reshape",
    "flatten",
    "permute",
    "transpose",
    "contiguous",
    "squeeze",
    "unsqueeze",
    "expand",
    "expand_as",
    "detach",
    "clone",
    "unbind",
    "split",
    "chunk",
    "__getitem__",
}

_COMPUTE_METHODS = {
    "add",
    "sub",
    "mul",
    "div",
    "matmul",
    "mm",
    "bmm",
    "sum",
    "mean",
    "amax",
    "amin",
    "max",
    "min",
    "abs",
    "neg",
    "exp",
    "exp2",
    "log",
    "log2",
    "sqrt",
    "rsqrt",
    "tanh",
    "floor",
    "ceil",
    "clamp",
}

_ACTIVATION_METHOD_POLICIES: Mapping[str, NodePolicy] = {
    "relu": NodePolicy(NodeRole.ACTIVATION, True, "relu"),
    "relu_": NodePolicy(NodeRole.ACTIVATION, True, "relu"),
    "relu6": NodePolicy(NodeRole.ACTIVATION, True, "relu6"),
    "softmax": NodePolicy(NodeRole.ACTIVATION, True, "softmax"),
    "sigmoid": NodePolicy(NodeRole.ACTIVATION, True, "sigmoid"),
    "gelu": NodePolicy(NodeRole.ACTIVATION),
    "silu": NodePolicy(NodeRole.ACTIVATION),
    "hardswish": NodePolicy(NodeRole.ACTIVATION),
    "hardsigmoid": NodePolicy(NodeRole.ACTIVATION, True, "hardsigmoid"),
}


def _coerce_policy(value: PolicyLike) -> NodePolicy:
    if isinstance(value, NodePolicy):
        return value
    if isinstance(value, NodeRole):
        return NodePolicy(value)
    try:
        return NodePolicy(NodeRole(value))
    except ValueError as exc:
        valid = ", ".join(role.value for role in NodeRole)
        raise ValueError(f"Unknown node role {value!r}; expected one of: {valid}") from exc


def _iter_fx_nodes(value: Any) -> Iterable[fx.Node]:
    if isinstance(value, fx.Node):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_fx_nodes(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_fx_nodes(item)
    elif isinstance(value, slice):
        yield from _iter_fx_nodes(value.start)
        yield from _iter_fx_nodes(value.stop)
        yield from _iter_fx_nodes(value.step)


def _target_label(target: Any) -> str:
    return getattr(target, "__qualname__", getattr(target, "__name__", str(target)))


def _meta_is_non_tensor(node: fx.Node) -> bool:
    if "val" in node.meta:
        value = node.meta["val"]
        flat_values = value if isinstance(value, (tuple, list)) else (value,)
        return not any(isinstance(item, torch.Tensor) for item in flat_values)
    value_type = node.meta.get("type")
    return value_type in {int, float, bool, str, torch.Size}


class ActivationStagePlanner:
    """Create strict producer-output activation boundaries for an FX graph."""

    def __init__(
        self,
        model: nn.Module | fx.GraphModule,
        *,
        additional_module_roles: Mapping[type[nn.Module] | str, PolicyLike] | None = None,
        additional_function_roles: Mapping[Callable[..., Any] | str, PolicyLike] | None = None,
        additional_method_roles: Mapping[str, PolicyLike] | None = None,
    ) -> None:
        self.graph_module = self._as_graph_module(model)
        self.modules = dict(self.graph_module.named_modules())
        self.additional_module_roles = {
            key: _coerce_policy(value)
            for key, value in (additional_module_roles or {}).items()
        }
        self.additional_function_roles = {
            key: _coerce_policy(value)
            for key, value in (additional_function_roles or {}).items()
        }
        self.additional_method_roles = {
            key: _coerce_policy(value)
            for key, value in (additional_method_roles or {}).items()
        }
        self._policies: dict[str, NodePolicy] = {}
        self._stages: list[ActivationStage] = []
        self._node_to_stage: dict[str, str] = {}
        self._topological_index = {
            node.name: index for index, node in enumerate(self.graph_module.graph.nodes)
        }

    @staticmethod
    def _as_graph_module(model: nn.Module | fx.GraphModule) -> fx.GraphModule:
        if isinstance(model, fx.GraphModule):
            return model
        if not isinstance(model, nn.Module):
            raise TypeError(f"Expected nn.Module or GraphModule, got {type(model).__name__}")
        if isinstance(model, nn.MultiheadAttention) or type(model).__name__.lower() in {
            "quantmha",
            "quantmultiheadattention",
        }:
            raise UnsupportedNodeError(
                f"Opaque multi-head attention {type(model).__name__} must be lowered "
                "to Q/K/V projections, score MatMul, Softmax, and value MatMul "
                "before activation transport"
            )
        try:
            from ..utils.fx_trace_utils import trace_quant_aware

            _, _, graph_module = trace_quant_aware(model)
            return graph_module
        except Exception as exc:
            raise ActivationStagePlanningError(
                f"FX tracing failed for {type(model).__name__}: {type(exc).__name__}: {exc}"
            ) from exc

    def plan(self) -> ActivationStagePlan:
        self._classify_graph()
        self._build_stages()
        node_sources = {
            node.name: self._source_stage_ids(node)
            for node in self.graph_module.graph.nodes
        }
        output_node = next(
            node for node in self.graph_module.graph.nodes if node.op == "output"
        )
        model_output_sources = self._sources_from_values((output_node.args, output_node.kwargs))
        stages = self._add_routes_to_stages()
        return ActivationStagePlan(
            graph_module=self.graph_module,
            stages=tuple(stages),
            node_to_stage=dict(self._node_to_stage),
            node_roles={name: policy.role for name, policy in self._policies.items()},
            node_sources=node_sources,
            model_output_sources=model_output_sources,
        )

    def _classify_graph(self) -> None:
        for node in self.graph_module.graph.nodes:
            policy = self._classify_node(node)
            self._policies[node.name] = policy

    def _classify_node(self, node: fx.Node) -> NodePolicy:
        if node.op == "placeholder":
            return NodePolicy(NodeRole.INPUT)
        if node.op == "get_attr":
            return NodePolicy(NodeRole.CONSTANT)
        if node.op == "output":
            return NodePolicy(NodeRole.OUTPUT)
        if node.op == "call_module":
            return self._classify_module(node)
        if node.op == "call_function":
            return self._classify_function(node)
        if node.op == "call_method":
            return self._classify_method(node)
        raise UnsupportedNodeError(
            f"Unsupported FX node kind {node.op!r} at {node.format_node()}"
        )

    def _classify_module(self, node: fx.Node) -> NodePolicy:
        module = self.modules.get(str(node.target))
        if module is None:
            raise ActivationStagePlanningError(
                f"FX node {node.name!r} references missing module {node.target!r}"
            )

        class_name = type(module).__name__
        qualified_name = f"{type(module).__module__}.{class_name}"
        for key, policy in self.additional_module_roles.items():
            if isinstance(key, str) and key in {class_name, qualified_name}:
                return policy
            if isinstance(key, type) and isinstance(module, key):
                return policy

        if isinstance(module, nn.MultiheadAttention) or class_name.lower() in {
            "quantmha",
            "quantmultiheadattention",
        }:
            raise UnsupportedNodeError(
                f"Opaque multi-head attention {qualified_name} at FX node "
                f"{node.name!r} must be lowered to Q/K/V projections, score "
                "MatMul, Softmax, and value MatMul before activation transport"
            )

        for module_type, policy in _ACTIVATION_MODULE_POLICIES:
            if isinstance(module, module_type):
                return policy
        named_policy = _ACTIVATION_NAME_POLICIES.get(class_name.lower())
        if named_policy is not None:
            return named_policy

        if isinstance(module, _TRANSPARENT_MODULE_TYPES) or class_name.lower() in _TRANSPARENT_MODULE_NAMES:
            if isinstance(module, _DROPOUT_MODULE_TYPES) and module.training:
                return NodePolicy(NodeRole.COMPUTE)
            return NodePolicy(NodeRole.TRANSPARENT)

        if isinstance(module, _COMPUTE_MODULE_TYPES) or class_name.lower() in _QUANT_COMPUTE_NAMES:
            return NodePolicy(NodeRole.COMPUTE)

        if _meta_is_non_tensor(node):
            return NodePolicy(NodeRole.NON_TENSOR)
        raise UnsupportedNodeError(
            f"Unhandled tensor-producing module {qualified_name} at FX node {node.name!r}; "
            "register it with additional_module_roles"
        )

    def _classify_function(self, node: fx.Node) -> NodePolicy:
        target = node.target
        target_name = _target_label(target)
        for key, policy in self.additional_function_roles.items():
            if key is target or (isinstance(key, str) and key == target_name):
                return policy

        policy = _ACTIVATION_FUNCTION_POLICIES.get(target)
        if policy is not None:
            return policy

        if target is builtins.getattr:
            attribute = node.args[1] if len(node.args) > 1 else None
            if attribute in {"shape", "ndim", "dtype", "device"}:
                return NodePolicy(NodeRole.NON_TENSOR)

        if target in _DROPOUT_FUNCTIONS or target in _STOCHASTIC_DEPTH_FUNCTIONS:
            training = node.kwargs.get("training")
            training_index = 3 if target in _STOCHASTIC_DEPTH_FUNCTIONS else 2
            if training is None and len(node.args) > training_index:
                training = node.args[training_index]
            if training is None:
                training = True
            if isinstance(training, fx.Node):
                raise UnsupportedNodeError(
                    f"Dropout training mode is data-dependent at FX node {node.name!r}"
                )
            return NodePolicy(NodeRole.COMPUTE if training else NodeRole.TRANSPARENT)

        if target is operator.getitem:
            inputs = tuple(_iter_fx_nodes(node.args[:1]))
            if inputs and self._policies[inputs[0].name].role == NodeRole.NON_TENSOR:
                return NodePolicy(NodeRole.NON_TENSOR)
            return NodePolicy(NodeRole.TRANSPARENT)

        if target in _TRANSPARENT_FUNCTIONS:
            return NodePolicy(NodeRole.TRANSPARENT)

        if target is getattr(F, "scaled_dot_product_attention", None):
            raise UnsupportedNodeError(
                f"Fused scaled-dot-product attention at FX node {node.name!r} must be "
                "lowered to score MatMul -> Softmax -> value MatMul before planning"
            )

        input_nodes = tuple(_iter_fx_nodes((node.args, node.kwargs)))
        if input_nodes and all(
            self._policies[input_node.name].role == NodeRole.NON_TENSOR
            for input_node in input_nodes
        ):
            return NodePolicy(NodeRole.NON_TENSOR)
        if target in _COMPUTE_FUNCTIONS:
            return NodePolicy(NodeRole.COMPUTE)
        if _meta_is_non_tensor(node):
            return NodePolicy(NodeRole.NON_TENSOR)
        raise UnsupportedNodeError(
            f"Unhandled tensor-producing function {target_name} at FX node {node.name!r}; "
            "register it with additional_function_roles"
        )

    def _classify_method(self, node: fx.Node) -> NodePolicy:
        target = str(node.target)
        if target in self.additional_method_roles:
            return self.additional_method_roles[target]
        policy = _ACTIVATION_METHOD_POLICIES.get(target)
        if policy is not None:
            return policy
        if target in _NON_TENSOR_METHODS:
            return NodePolicy(NodeRole.NON_TENSOR)
        if target in _TRANSPARENT_METHODS:
            inputs = tuple(_iter_fx_nodes(node.args[:1]))
            if inputs and self._policies[inputs[0].name].role == NodeRole.NON_TENSOR:
                return NodePolicy(NodeRole.NON_TENSOR)
            return NodePolicy(NodeRole.TRANSPARENT)
        input_nodes = tuple(_iter_fx_nodes((node.args, node.kwargs)))
        if input_nodes and all(
            self._policies[input_node.name].role == NodeRole.NON_TENSOR
            for input_node in input_nodes
        ):
            return NodePolicy(NodeRole.NON_TENSOR)
        if target in _COMPUTE_METHODS:
            return NodePolicy(NodeRole.COMPUTE)
        if _meta_is_non_tensor(node):
            return NodePolicy(NodeRole.NON_TENSOR)
        raise UnsupportedNodeError(
            f"Unhandled tensor-producing method {target!r} at FX node {node.name!r}; "
            "register it with additional_method_roles"
        )

    def _build_stages(self) -> None:
        for node in self.graph_module.graph.nodes:
            policy = self._policies[node.name]
            if policy.role == NodeRole.INPUT:
                self._new_stage(node, StageKind.INPUT, policy)
            elif policy.role == NodeRole.COMPUTE:
                self._new_stage(node, StageKind.COMPUTE, policy)
            elif policy.role == NodeRole.ACTIVATION:
                fused = self._try_fuse_activation(node, policy)
                if not fused:
                    self._new_stage(node, StageKind.ACTIVATION, policy)

    def _new_stage(self, node: fx.Node, kind: StageKind, policy: NodePolicy) -> None:
        stage_id = f"stage_{len(self._stages):04d}"
        inputs = () if kind == StageKind.INPUT else self._sources_from_values((node.args, node.kwargs))
        stage = ActivationStage(
            stage_id=stage_id,
            kind=kind,
            node_names=(node.name,),
            entry_node=node.name,
            output_node=node.name,
            input_stage_ids=inputs,
            is_unsigned=policy.is_unsigned,
            unsigned_source=policy.unsigned_source,
        )
        self._stages.append(stage)
        self._node_to_stage[node.name] = stage_id

    def _try_fuse_activation(self, node: fx.Node, policy: NodePolicy) -> bool:
        tensor_inputs = self._tensor_input_nodes(node)
        if len(tensor_inputs) != 1:
            return False

        cursor = tensor_inputs[0]
        downstream = node
        transparent_chain: list[fx.Node] = []
        while self._policies[cursor.name].role == NodeRole.TRANSPARENT:
            if self._value_users(cursor) != (downstream,):
                return False
            predecessors = self._tensor_input_nodes(cursor)
            if len(predecessors) != 1:
                return False
            transparent_chain.append(cursor)
            downstream = cursor
            cursor = predecessors[0]

        stage_id = self._node_to_stage.get(cursor.name)
        if stage_id is None or self._value_users(cursor) != (downstream,):
            return False
        stage_index = next(
            index for index, stage in enumerate(self._stages) if stage.stage_id == stage_id
        )
        stage = self._stages[stage_index]
        if stage.kind == StageKind.INPUT or stage.output_node != cursor.name:
            return False

        chain = tuple(item.name for item in reversed(transparent_chain))
        fused_names = stage.node_names + chain + (node.name,)
        self._stages[stage_index] = replace(
            stage,
            node_names=fused_names,
            output_node=node.name,
            is_unsigned=policy.is_unsigned,
            unsigned_source=policy.unsigned_source,
        )
        for fused_name in chain + (node.name,):
            self._node_to_stage[fused_name] = stage_id
        return True

    def _tensor_input_nodes(self, node: fx.Node) -> tuple[fx.Node, ...]:
        inputs = []
        seen = set()
        for input_node in _iter_fx_nodes((node.args, node.kwargs)):
            role = self._policies[input_node.name].role
            if role in {NodeRole.CONSTANT, NodeRole.NON_TENSOR}:
                continue
            if input_node.name not in seen:
                seen.add(input_node.name)
                inputs.append(input_node)
        return tuple(inputs)

    def _value_users(self, node: fx.Node) -> tuple[fx.Node, ...]:
        users = [
            user
            for user in node.users
            if self._policies[user.name].role != NodeRole.NON_TENSOR
        ]
        users.sort(key=lambda user: self._topological_index[user.name])
        return tuple(users)

    def _source_stage_ids(self, node: fx.Node) -> tuple[str, ...]:
        stage_id = self._node_to_stage.get(node.name)
        if stage_id is not None:
            return (stage_id,)
        role = self._policies[node.name].role
        if role in {NodeRole.CONSTANT, NodeRole.NON_TENSOR}:
            return ()
        return self._sources_from_values((node.args, node.kwargs))

    def _sources_from_values(self, values: Any) -> tuple[str, ...]:
        sources = []
        seen = set()
        for input_node in _iter_fx_nodes(values):
            for stage_id in self._source_stage_ids(input_node):
                if stage_id not in seen:
                    seen.add(stage_id)
                    sources.append(stage_id)
        return tuple(sources)

    def _add_routes_to_stages(self) -> list[ActivationStage]:
        nodes = {node.name: node for node in self.graph_module.graph.nodes}
        routed_stages = []
        for stage in self._stages:
            consumers: set[str] = set()
            passthrough: set[str] = set()
            has_fanout = False

            def visit(current: fx.Node) -> None:
                nonlocal has_fanout
                value_users = self._value_users(current)
                if len(value_users) > 1:
                    has_fanout = True
                for user in value_users:
                    role = self._policies[user.name].role
                    if role == NodeRole.OUTPUT:
                        continue
                    owner = self._node_to_stage.get(user.name)
                    if role == NodeRole.TRANSPARENT and owner is None:
                        if user.name in passthrough:
                            continue
                        passthrough.add(user.name)
                        visit(user)
                    elif owner != stage.stage_id:
                        consumers.add(user.name)

            visit(nodes[stage.output_node])
            ordered_consumers = tuple(
                sorted(consumers, key=self._topological_index.__getitem__)
            )
            ordered_passthrough = tuple(
                sorted(passthrough, key=self._topological_index.__getitem__)
            )
            consumer_stage_ids = []
            for consumer in ordered_consumers:
                consumer_stage_id = self._node_to_stage.get(consumer)
                if consumer_stage_id is not None and consumer_stage_id not in consumer_stage_ids:
                    consumer_stage_ids.append(consumer_stage_id)
            routed_stages.append(
                replace(
                    stage,
                    consumer_nodes=ordered_consumers,
                    consumer_stage_ids=tuple(consumer_stage_ids),
                    passthrough_nodes=ordered_passthrough,
                    has_fanout=has_fanout,
                )
            )
        return routed_stages


def plan_activation_stages(
    model: nn.Module | fx.GraphModule,
    *,
    additional_module_roles: Mapping[type[nn.Module] | str, PolicyLike] | None = None,
    additional_function_roles: Mapping[Callable[..., Any] | str, PolicyLike] | None = None,
    additional_method_roles: Mapping[str, PolicyLike] | None = None,
) -> ActivationStagePlan:
    """Plan producer-output hardware activation boundaries for ``model``."""

    return ActivationStagePlanner(
        model,
        additional_module_roles=additional_module_roles,
        additional_function_roles=additional_function_roles,
        additional_method_roles=additional_method_roles,
    ).plan()


__all__ = [
    "ActivationStage",
    "ActivationStagePlan",
    "ActivationStagePlanner",
    "ActivationStagePlanningError",
    "NodePolicy",
    "NodeRole",
    "StageKind",
    "UnsupportedNodeError",
    "plan_activation_stages",
]
