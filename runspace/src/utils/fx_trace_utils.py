import torch
import torch.fx
import operator as _operator
import math

try:
    from ..registry.op_registry import OpRegistry
    from ..ops.quant_mha import (
        DecomposedMultiheadAttention,
        DecomposedQkvAttention,
        DecomposedMlpBlock,
        ScaledDotProduct,
        AttentionWeightedValues,
    )
except ImportError:
    from src.registry.op_registry import OpRegistry
    from src.ops.quant_mha import (
        DecomposedMultiheadAttention,
        DecomposedQkvAttention,
        DecomposedMlpBlock,
        ScaledDotProduct,
        AttentionWeightedValues,
    )


class FalseBoolProxy(torch.fx.Proxy):
    """Proxy that resolves boolean checks as False during symbolic tracing.

    This keeps tracing on the standard fixed-shape inference path for models
    like timm MobileViT that guard reshape/interpolate branches with shape
    comparisons.
    """

    def __bool__(self):
        return False


class QuantAwareTracer(torch.fx.Tracer):
    """Shared FX tracer used by graphing and runtime functional-op replacement."""

    def proxy(self, node):
        return FalseBoolProxy(node, self)

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        # Use class names for comparison to avoid issues with multiple class definitions
        # across different import paths (e.g. src.ops vs runspace.src.ops)
        cls_name = m.__class__.__name__
        if cls_name in (
            "DecomposedMultiheadAttention", 
            "DecomposedQkvAttention", 
            "DecomposedMlpBlock",
            "ScaledDotProduct",
            "AttentionWeightedValues",
        ):
            return False

        quantized_ops = tuple(
            dict.fromkeys(
                list(OpRegistry.get_supported_ops().values())
                + list(getattr(OpRegistry, "_registry", {}).values())
            )
        )
        if isinstance(m, quantized_ops):
            return True

        return super().is_leaf_module(m, module_qualified_name)


_INT_ARITH_OPS = {
    _operator.add, _operator.sub, _operator.mul,
    _operator.truediv, _operator.floordiv, _operator.mod, _operator.pow,
    _operator.rshift, _operator.lshift,
    _operator.neg, _operator.abs,
    _operator.getitem,
    _operator.and_, _operator.or_, _operator.xor,
    torch.add, torch.sub, torch.mul, torch.div,
    math.ceil, math.floor,
}


def find_non_tensor_nodes(graph: torch.fx.Graph) -> set:
    """Forward-propagate a non-tensor label through shape/integer arithmetic."""
    non_tensor = set()

    for node in graph.nodes:
        if node.op == "call_function":
            target = node.target

            if target is getattr and len(node.args) >= 2 and node.args[1] == "shape":
                non_tensor.add(node.name)
                continue

            if target in _INT_ARITH_OPS:
                node_inputs = [arg for arg in node.args if isinstance(arg, torch.fx.Node)]
                if node_inputs and all(arg.name in non_tensor for arg in node_inputs):
                    non_tensor.add(node.name)

        elif node.op == "call_method":
            if node.target in {"size", "dim", "numel", "stride", "item", "tolist"}:
                non_tensor.add(node.name)

    return non_tensor


def trace_quant_aware(model: torch.nn.Module):
    """Trace a model with quant-aware leaf handling and MobileViT-safe bool semantics."""
    tracer = QuantAwareTracer()
    graph = tracer.trace(model)
    return tracer, graph, torch.fx.GraphModule(model, graph)


def fx_trace_subtrees(model, _path: str = '', _depth: int = 0, _max_depth: int = 6):
    """Yield (qualified_path, GraphModule) pairs for the largest traceable
    subtrees of `model`. Tries to trace at the current level first; on
    TraceError (or any other failure) descends into named children.

    Raises RuntimeError if a leaf module fails to trace, or _max_depth is
    exceeded — the caller can only certify coverage if every op is
    FX-visible (or wrapped in @observer, which makes it FX-opaque). No
    silent fallback at this level; callers may catch and degrade if they
    have a meaningful fallback.
    """
    try:
        _, _, traced = trace_quant_aware(model)
        yield _path, traced
        return
    except Exception as e:
        children = list(model.named_children())
        if not children:
            raise RuntimeError(
                f"FX trace failed at leaf module {_path!r}: {type(e).__name__}: {e}"
            ) from e
        if _depth >= _max_depth:
            raise RuntimeError(
                f"FX trace failed and max descent depth {_max_depth} reached at {_path!r}: "
                f"{type(e).__name__}: {e}"
            ) from e
        for child_name, child in children:
            sub_path = f'{_path}.{child_name}' if _path else child_name
            yield from fx_trace_subtrees(child, sub_path, _depth + 1, _max_depth)
