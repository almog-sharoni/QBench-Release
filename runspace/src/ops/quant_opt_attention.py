"""
Quantized OPT attention.

HuggingFace OPT computes attention scores with two functional ``torch.bmm``
calls and a functional softmax inside ``OPTAttention.forward`` (when loaded with
``attn_implementation="eager"``). Those functional ops are invisible to the
module-swap quantization path, so the attention-score compute would otherwise
stay in FP32 while only the q/k/v/out projections are quantized.

``QuantOPTAttention`` subclasses ``OPTAttention`` and overrides ``forward`` with
the upstream (transformers 4.46.3) body, routing the QKᵀ matmul, the softmax,
and the attention·V matmul through real quantized modules (``QuantBMM`` /
``QuantSoftmax``). Because these are registered quantized ops, they also show up
in the coverage / compliance report.

The projection layers (q/k/v/out_proj) are left untouched — the GenericAdapter's
recursive replace has already swapped them to ``QuantLinear`` before an instance
is converted here.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from transformers.models.opt.modeling_opt import OPTAttention

from .quant_matmul import QuantBMM
from .quant_softmax import QuantSoftmax


class QuantOPTAttention(OPTAttention):
    # HF masks padded/causal positions by ADDING finfo.min (~-3.4e38) to the
    # logits. A per-tensor-quantized softmax would let that giant value set the
    # quantization scale and flush every real logit to zero. We instead clamp
    # masked logits to a representable floor before the quantized softmax:
    # exp(floor - rowmax) is still ~0 so causality is preserved, but the scale
    # is governed by the real logits (~O(10)).
    softmax_mask_floor = -30.0

    @classmethod
    def convert(cls, module: OPTAttention, q_type: str = "fp8_e4m3",
                quant_mode: str = "tensor", chunk_size=None,
                layer_name: str = "", run_id: str = "default") -> "QuantOPTAttention":
        """In-place class swap of an existing OPTAttention, attaching the
        quantized score-path sub-ops. Preserves the already-quantized
        q/k/v/out_proj children."""
        module.__class__ = cls

        def _mk_bmm():
            op = QuantBMM(q_type=q_type, quant_mode=quant_mode, chunk_size=chunk_size)
            # QuantBMM defaults input_quantization=False (pass-through); turn it
            # on so both operands are actually quantized, and pin per-operand
            # formats so the compliance table reports them.
            op.input_quantization = True
            op.input1_q_type = q_type
            op.input2_q_type = q_type
            op.weight_quantization = False
            op.output_quantization = False
            op.layer_name = layer_name
            op.run_id = run_id
            return op

        module.qk_bmm = _mk_bmm()           # QKᵀ
        module.av_bmm = _mk_bmm()           # attn_probs · V
        softmax = QuantSoftmax(dim=-1, q_type=q_type, quant_mode=quant_mode,
                               chunk_size=chunk_size)
        softmax.input_quantization = True
        softmax.output_quantization = False
        softmax.layer_name = layer_name
        softmax.run_id = run_id
        module.attn_softmax = softmax
        return module

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scaling
        if is_cross_attention and past_key_value is not None:
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        # QKᵀ — quantized batch matmul (was torch.bmm)
        attn_weights = self.qk_bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            # Clamp masked logits to a quantization-representable floor (instead
            # of HF's finfo.min) so the quantized softmax's per-tensor scale is
            # not blown out by ~-3.4e38.
            attn_weights = torch.clamp(attn_weights, min=self.softmax_mask_floor)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # softmax — quantized (was nn.functional.softmax). FP16 upcast path
        # falls back to the stock functional softmax to preserve dtype handling.
        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = self.attn_softmax(attn_weights)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # attn_probs · V — quantized batch matmul (was torch.bmm)
        attn_output = self.av_bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value
