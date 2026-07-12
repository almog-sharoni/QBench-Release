import os
import operator
import types
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .generic_adapter import GenericAdapter


_ACTIVATION_TRACE_INPUT_NAMES = ("input_ids", "attention_mask")
_ACTIVATION_METADATA_INPUT_NAMES = {
    "input_ids",
    "attention_mask",
    "position_ids",
    "token_type_ids",
    "head_mask",
}
_ACTIVATION_BYPASS_META_KEY = "qbench_activation_bypass"


def _iter_fx_nodes(value):
    if isinstance(value, torch.fx.Node):
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


def _is_auxiliary_fx_node(node: torch.fx.Node) -> bool:
    """Return whether an FX node carries shape/device/dtype metadata."""
    if node.op == "get_attr":
        return True
    if node.op == "call_method" and node.target in {
        "size",
        "dim",
        "ndimension",
        "numel",
        "stride",
        "item",
        "tolist",
        "__len__",
    }:
        return True
    if node.op == "call_function":
        if node.target is getattr:
            attribute = node.args[1] if len(node.args) > 1 else None
            return attribute in {"shape", "ndim", "dtype", "device"}
        if node.target is torch.finfo:
            return True
        if node.target is operator.getitem:
            inputs = tuple(_iter_fx_nodes(node.args[:1]))
            return bool(inputs) and all(
                bool(input_node.meta.get("qbench_slm_auxiliary"))
                for input_node in inputs
            )
    return False


def _mark_activation_metadata_bypasses(graph_module: torch.fx.GraphModule) -> None:
    """Tag token/mask-derived values that are hardware metadata, not activations."""
    modules = dict(graph_module.named_modules())
    metadata_nodes = set()
    auxiliary_nodes = set()

    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            if str(node.target) in _ACTIVATION_METADATA_INPUT_NAMES:
                metadata_nodes.add(node.name)
                node.meta[_ACTIVATION_BYPASS_META_KEY] = "slm_model_metadata"
            continue
        if node.op == "output":
            continue

        if _is_auxiliary_fx_node(node):
            auxiliary_nodes.add(node.name)
            node.meta["qbench_slm_auxiliary"] = True
            continue

        input_nodes = tuple(_iter_fx_nodes((node.args, node.kwargs)))
        has_activation_input = any(
            input_node.name not in metadata_nodes
            and input_node.name not in auxiliary_nodes
            for input_node in input_nodes
        )
        has_metadata_input = any(
            input_node.name in metadata_nodes for input_node in input_nodes
        )

        produces_activation = False
        if node.op == "call_module":
            produces_activation = isinstance(
                modules.get(str(node.target)),
                nn.Embedding,
            )
        elif node.op == "call_function":
            produces_activation = node.target is F.embedding

        if produces_activation or has_activation_input:
            continue
        if has_metadata_input:
            metadata_nodes.add(node.name)
            node.meta[_ACTIVATION_BYPASS_META_KEY] = "slm_model_metadata"
        else:
            auxiliary_nodes.add(node.name)
            node.meta["qbench_slm_auxiliary"] = True


def _slm_activation_planner_kwargs() -> dict:
    """Policies needed for the operations emitted by Transformers' OPT tracer."""
    return {
        "additional_module_roles": {"Embedding": "compute"},
        "additional_function_roles": {
            torch.cumsum: "transparent",
            F.embedding: "compute",
            torch.max: "transparent",
        },
        "additional_method_roles": {
            "to": "transparent",
            "masked_fill": "transparent",
            "masked_fill_": "transparent",
            "bool": "transparent",
            "long": "transparent",
        },
    }


def _slm_activation_trace_provider(model: nn.Module):
    """Trace an HF causal LM for producer-stage activation transport."""
    try:
        from transformers.utils.fx import HFTracer, symbolic_trace
    except ImportError as exc:
        raise RuntimeError(
            "SLM activation transport requires transformers with its FX tracer"
        ) from exc

    class QBenchSLMTracer(HFTracer):
        def is_leaf_module(self, module: nn.Module, module_qualified_name: str) -> bool:
            class_name = type(module).__name__
            if class_name == "QuantOPTAttention":
                return False
            if class_name.startswith("Quant"):
                return True
            return super().is_leaf_module(module, module_qualified_name)

        def to_bool(self, _proxy) -> bool:
            # QBench Quant* wrappers contain validation-only shape/dtype branches.
            # Their inference path is the false branch, matching QuantAwareTracer.
            return False

    config = getattr(model, "config", None)
    original_use_cache = getattr(config, "use_cache", None)
    if config is not None and original_use_cache is not None:
        config.use_cache = False
    try:
        graph_module = symbolic_trace(
            model,
            input_names=list(_ACTIVATION_TRACE_INPUT_NAMES),
            disable_check=True,
            tracer_cls=QBenchSLMTracer,
        )
    except Exception as exc:
        raise RuntimeError(
            "SLM activation transport could not trace the HuggingFace model. "
            "Only architectures with an explicit QBench SLM trace policy are "
            f"supported; {type(model).__name__} failed with "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if config is not None and original_use_cache is not None:
            config.use_cache = original_use_cache

    _mark_activation_metadata_bypasses(graph_module)
    return graph_module, _slm_activation_planner_kwargs()


def _install_activation_trace_provider(model: nn.Module) -> None:
    model._qbench_activation_trace_provider = types.MethodType(
        _slm_activation_trace_provider,
        model,
    )


@dataclass(frozen=True)
class SLMArchitectureWrapper:
    """Architecture-specific wrapper hook for functional decoder internals."""

    name: str
    description: str
    loader: Callable[[], tuple[Callable[[nn.Module], bool], Callable[[nn.Module, dict], None]]]


def _load_opt_attention_wrapper():
    """Return match/convert callbacks for HuggingFace OPT attention."""
    from transformers.models.opt.modeling_opt import OPTAttention
    from ..ops.quant_opt_attention import QuantOPTAttention

    def matches(module: nn.Module) -> bool:
        return isinstance(module, OPTAttention) and not isinstance(module, QuantOPTAttention)

    def convert(module: nn.Module, quant_kwargs: dict) -> None:
        QuantOPTAttention.convert(module, **quant_kwargs)

    return matches, convert


SLM_ARCHITECTURE_WRAPPERS = (
    SLMArchitectureWrapper(
        name="opt_attention",
        description="OPTAttention score path: qk bmm, softmax, and attn-value bmm",
        loader=_load_opt_attention_wrapper,
    ),
)


class SLMAdapter(GenericAdapter):
    """
    Adapter for Small Language Models (decoder-only causal LMs) loaded from
    HuggingFace ``transformers``.

    It reuses GenericAdapter's recursive layer-replacement machinery so the
    existing PyTorch quant ops apply unchanged: ``nn.Linear`` -> ``QuantLinear``,
    ``nn.LayerNorm`` -> ``QuantLayerNorm``, ``nn.ReLU`` -> ``QuantReLU``, etc.

    What differs from the vision path:
      * the base model is loaded via ``AutoModelForCausalLM`` instead of
        torchvision/timm,
      * Conv+BN folding and input-normalization folding are disabled (no Conv
        stem to fold), and
      * the generic vision FX rewrite is disabled because its tracer and dummy
        input assume image models. Producer-stage activation transport instead
        uses the HuggingFace-aware trace provider installed on the model.

    Batches are expected to already be tokenized into fixed-length blocks of
    token ids (see ``wikitext2_lm`` dataset). The model is evaluated as a causal
    LM and scored with perplexity via the shared ``MetricsEngine`` (its 3-D
    logits branch).
    """

    def __init__(self, *args, **kwargs):
        # SLMs have no Conv/BN stem or image input normalization. The generic FX
        # rewrite is vision-specific; activation transport uses the dedicated
        # HuggingFace trace provider installed by build_model().
        kwargs["fold_layers"] = False
        kwargs["fold_input_norm"] = False
        kwargs["enable_fx_quantization"] = False
        # Disable timm vision heuristics (qkv / fc1+fc2 attribute matching) that
        # misfire on HF decoder layers and would replace whole transformer blocks.
        self._enable_timm_decomposition = False
        # Default to HF source so model_source='auto' doesn't try torchvision/timm.
        if kwargs.get("model_source", "auto") in (None, "auto"):
            kwargs["model_source"] = "huggingface"
        super().__init__(*args, **kwargs)

    def _load_base_model(self) -> nn.Module:
        """Load a decoder-only causal LM from HuggingFace transformers."""
        if self.base_model_instance is not None:
            import copy
            return copy.deepcopy(self.base_model_instance)

        from transformers import AutoModelForCausalLM

        # `weights` may point at a local checkpoint directory; otherwise the
        # model name is treated as a HuggingFace repo id (e.g. facebook/opt-125m).
        pretrained = self.model_name
        if isinstance(self.weights, str) and self.weights.strip() and os.path.isdir(self.weights):
            pretrained = self.weights

        print(f"SLMAdapter: loading causal LM '{pretrained}' from transformers...")
        # attn_implementation="eager" expresses attention as explicit
        # torch.bmm + softmax ops (instead of the fused SDPA kernel), so the
        # attention-score compute is visible to the quantization machinery
        # instead of being sealed inside scaled_dot_product_attention.
        model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
        model.eval()
        return model

    def prepare_batch(self, batch):
        """
        Convert a dataloader batch into ``(model_inputs, labels)``.

        Accepts either a bare ``input_ids`` tensor or a dict containing an
        ``input_ids`` (and optional ``attention_mask`` / ``labels``) key. Labels
        default to the input ids. Token IDs and masks remain integer metadata;
        activation transport starts at the embedding outputs.
        """
        if isinstance(batch, dict):
            input_ids = batch["input_ids"]
            labels = batch.get("labels", input_ids)
            attention_mask = batch.get("attention_mask")
        elif isinstance(batch, (list, tuple)):
            input_ids = batch[0]
            labels = batch[1] if len(batch) > 1 else batch[0]
            attention_mask = None
        else:
            input_ids = batch
            labels = batch
            attention_mask = None

        if not torch.is_tensor(input_ids):
            input_ids = torch.as_tensor(input_ids)
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)
        input_ids = input_ids.long()
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        elif not torch.is_tensor(attention_mask):
            attention_mask = torch.as_tensor(attention_mask)
        attention_mask = attention_mask.long()
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "SLM attention_mask must have the same shape as input_ids; "
                f"got {tuple(attention_mask.shape)} and {tuple(input_ids.shape)}"
            )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }, labels.long()

    def forward(self, model: nn.Module, batch):
        """Run the causal-LM forward and return logits ``[B, seq, vocab]``."""
        model_inputs, _ = batch
        if isinstance(model_inputs, dict):
            outputs = model(**model_inputs)
        else:
            input_ids = model_inputs.long()
            outputs = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
            )
        if hasattr(outputs, "logits"):
            return outputs.logits
        if isinstance(outputs, dict) and "logits" in outputs:
            return outputs["logits"]
        return outputs

    architecture_wrappers = SLM_ARCHITECTURE_WRAPPERS

    def build_model(self, quantized: bool = False) -> nn.Module:
        """Build the model, then apply architecture-specific quant wrappers.

        HuggingFace decoder blocks commonly hide important compute in Python
        functional calls that are not FX-traceable in this adapter. The wrapper
        registry keeps those architecture-specific conversions explicit: each
        registered wrapper swaps a known HF module to a Quant* module that is
        composed from existing QBench ops. Unsupported architectures can still
        build, but their missing wrappers are reported by the SLM support probe.
        """
        model = super().build_model(quantized=quantized)
        if quantized:
            self._apply_architecture_wrappers(model)
        _install_activation_trace_provider(model)
        return model

    def _architecture_quant_kwargs(self, layer_name: str) -> dict:
        return {
            "q_type": self.quantization_type,
            "quant_mode": self.quant_mode,
            "chunk_size": self.input_chunk_size if self.input_chunk_size is not None else self.chunk_size,
            "layer_name": layer_name,
            "run_id": getattr(self, "run_id", "default"),
        }

    def _apply_architecture_wrappers(self, model: nn.Module):
        """Apply all registered architecture wrappers that match this model."""
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        self.applied_architecture_wrappers = {}
        self.unavailable_architecture_wrappers = {}

        for wrapper in self.architecture_wrappers:
            try:
                matches, convert = wrapper.loader()
            except Exception as exc:
                self.unavailable_architecture_wrappers[wrapper.name] = f"{type(exc).__name__}: {exc}"
                continue

            count = 0
            for name, module in list(model.named_modules()):
                if not matches(module):
                    continue
                convert(module, self._architecture_quant_kwargs(name))
                module.to(device)
                count += 1

            if count:
                self.applied_architecture_wrappers[wrapper.name] = count
                print(
                    f"SLMAdapter: applied {wrapper.name} to {count} modules "
                    f"({wrapper.description})."
                )

    def _quantize_attention(self, model: nn.Module):
        """Compatibility alias for older callers."""
        self._apply_architecture_wrappers(model)

    def build_reference_model(self) -> nn.Module:
        """Build an FP reference model (no quantization)."""
        return self._load_base_model()
