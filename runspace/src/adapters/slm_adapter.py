import os
import torch
import torch.nn as nn

from .generic_adapter import GenericAdapter


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
      * FX functional rewriting is disabled because HuggingFace decoder forwards
        are not ``torch.fx``-traceable (and the FX path's dummy-input/validation
        assumes an image tensor).

    Batches are expected to already be tokenized into fixed-length blocks of
    token ids (see ``wikitext2_lm`` dataset). The model is evaluated as a causal
    LM and scored with perplexity via the shared ``MetricsEngine`` (its 3-D
    logits branch).
    """

    def __init__(self, *args, **kwargs):
        # SLMs have no Conv/BN stem and no image input normalization to fold,
        # and HF decoder graphs aren't FX-traceable — force these off regardless
        # of what the config/defaults pass in.
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
        Convert a dataloader batch into ``(input_ids, labels)``.

        Accepts either a bare ``input_ids`` tensor or a dict containing an
        ``input_ids`` (and optional ``labels``) key. Labels default to the
        input ids — the MetricsEngine shifts internally for next-token scoring.
        """
        if isinstance(batch, dict):
            input_ids = batch["input_ids"]
            labels = batch.get("labels", input_ids)
        elif isinstance(batch, (list, tuple)):
            input_ids = batch[0]
            labels = batch[1] if len(batch) > 1 else batch[0]
        else:
            input_ids = batch
            labels = batch

        if not torch.is_tensor(input_ids):
            input_ids = torch.as_tensor(input_ids)
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)
        return input_ids.long(), labels.long()

    def forward(self, model: nn.Module, batch):
        """Run the causal-LM forward and return logits ``[B, seq, vocab]``."""
        input_ids, _ = batch
        outputs = model(input_ids=input_ids)
        return outputs.logits if hasattr(outputs, "logits") else outputs

    def build_model(self, quantized: bool = False) -> nn.Module:
        """Build the model, then (when quantized) also quantize the attention
        score path — the QKᵀ / softmax / attn·V ops that live as functional
        calls inside the HF attention module and so are missed by the
        module-swap recursive replace."""
        model = super().build_model(quantized=quantized)
        if quantized:
            self._quantize_attention(model)
        return model

    def _quantize_attention(self, model: nn.Module):
        """Swap each eager OPT attention module to QuantOPTAttention, attaching
        quantized bmm/softmax sub-ops configured from this adapter's settings."""
        try:
            from transformers.models.opt.modeling_opt import OPTAttention
        except Exception:
            return
        from ..ops.quant_opt_attention import QuantOPTAttention

        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        count = 0
        for name, module in model.named_modules():
            # Match eager OPTAttention but skip already-converted instances.
            if isinstance(module, OPTAttention) and not isinstance(module, QuantOPTAttention):
                QuantOPTAttention.convert(
                    module,
                    q_type=self.quantization_type,
                    quant_mode=self.quant_mode,
                    chunk_size=self.input_chunk_size if self.input_chunk_size is not None else self.chunk_size,
                    layer_name=name,
                    run_id=getattr(self, "run_id", "default"),
                )
                module.to(device)
                count += 1
        if count:
            print(f"SLMAdapter: quantized attention score path in {count} attention blocks "
                  f"(QKᵀ / softmax / attn·V -> QuantBMM/QuantSoftmax).")

    def build_reference_model(self) -> nn.Module:
        """Build an FP reference model (no quantization)."""
        return self._load_base_model()
