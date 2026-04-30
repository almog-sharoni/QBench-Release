import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

from runspace.src.registry.op_registry import OpRegistry
from runspace.src.ops.quant_base import QuantizedLayerMixin
from runspace.src.ops.quant_arithmetic import _QuantArithmeticBase


class ScaledDotProduct(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim

    def forward(self, q, k, attn_mask, key_padding_mask):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            scores = scores + attn_mask

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float('-inf'))

        return scores


class AttentionWeightedValues(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, attn_weights, v, batch_size, tgt_len):
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.embed_dim)
        return attn_output

@OpRegistry.register("DecomposedMultiheadAttention", original_cls=nn.MultiheadAttention) 
class DecomposedMultiheadAttention(_QuantArithmeticBase):
    """
    A decomposed MultiheadAttention module that uses nn.Linear for projections
    to allow for quantization of internal layers.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, q_type="fp8_e4m3", quantization_bias: int = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_type = q_type
        self.quantization_bias = quantization_bias

        # Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout_layer = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

        self.scaled_dot_product = ScaledDotProduct(self.head_dim)
        self.attention_weighted_values = AttentionWeightedValues(embed_dim)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None, average_attn_weights=True):
        # Assuming batch_first=True for simplicity as it's common in ViT
        # query, key, value shape: (batch_size, seq_len, embed_dim)
        
        batch_size, tgt_len, _ = query.shape
        src_len = key.shape[1]

        # Quantize inputs
        query, key, value = self._quantize_operands([query, key, value])
        
        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Match previous tensor flow exactly:
        # (B, L, E) -> (B, H, L, D)
        q = q.reshape(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = self.scaled_dot_product(q, k, attn_mask, key_padding_mask)
        
        attn_weights = self.softmax(scores)
        attn_weights = self.dropout_layer(attn_weights)
        
        attn_output = self.attention_weighted_values(attn_weights, v, batch_size, tgt_len)
        
        # Output projection
        attn_output = self.out_proj(attn_output)
        
        if need_weights:
            return attn_output, attn_weights
        else:
            return attn_output, None

    @classmethod
    def from_native(cls, native_mha: nn.MultiheadAttention, q_type="fp8_e4m3", quantization_bias: int = None):
        """Creates a DecomposedMultiheadAttention from a native nn.MultiheadAttention module."""
        decomposed = cls(
            embed_dim=native_mha.embed_dim,
            num_heads=native_mha.num_heads,
            dropout=native_mha.dropout,
            bias=native_mha.in_proj_bias is not None,
            q_type=q_type,
            quantization_bias=quantization_bias
        )
        
        # Copy weights
        # Native MHA packs in_proj_weight as [q_weight, k_weight, v_weight]
        if native_mha.in_proj_weight is not None:
            q_w, k_w, v_w = native_mha.in_proj_weight.chunk(3, dim=0)
            decomposed.q_proj.weight.data.copy_(q_w)
            decomposed.k_proj.weight.data.copy_(k_w)
            decomposed.v_proj.weight.data.copy_(v_w)
            
        if native_mha.in_proj_bias is not None:
            q_b, k_b, v_b = native_mha.in_proj_bias.chunk(3, dim=0)
            decomposed.q_proj.bias.data.copy_(q_b)
            decomposed.k_proj.bias.data.copy_(k_b)
            decomposed.v_proj.bias.data.copy_(v_b)
            
        decomposed.out_proj.weight.data.copy_(native_mha.out_proj.weight.data)
        if native_mha.out_proj.bias is not None:
            decomposed.out_proj.bias.data.copy_(native_mha.out_proj.bias.data)
            
        return decomposed


@OpRegistry.register("DecomposedQkvAttention")
class DecomposedQkvAttention(nn.Module, QuantizedLayerMixin):
    """
    Decomposed timm-style attention block (qkv + proj).
    Works with modules exposing: qkv, proj, num_heads and optional dropouts.
    """
    def __init__(self, embed_dim, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0, q_type="fp8_e4m3", quantization_bias=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.attn_dim = embed_dim

        self.scale = self.head_dim ** -0.5
        self.q_type = q_type
        self.quantization_bias = quantization_bias

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = nn.Identity()
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.fused_attn = False

    def forward(self, x, attn_mask=None, is_causal=False, **kwargs):
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.fused_attn:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            q = q * self.scale
            attn = torch.matmul(q, k.transpose(-2, -1))
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(bsz, seq_len, self.attn_dim)
        out = self.norm(out)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    @classmethod
    def from_native(cls, native_attn, q_type="fp8_e4m3", quantization_bias=None):
        if not (hasattr(native_attn, "qkv") and hasattr(native_attn, "proj") and hasattr(native_attn, "num_heads")):
            raise TypeError("Module does not look like a timm-style Attention block.")

        qkv = native_attn.qkv
        proj = native_attn.proj
        embed_dim = qkv.in_features

        attn_drop_p = native_attn.attn_drop.p if isinstance(getattr(native_attn, "attn_drop", None), nn.Dropout) else 0.0
        proj_drop_p = native_attn.proj_drop.p if isinstance(getattr(native_attn, "proj_drop", None), nn.Dropout) else 0.0

        decomposed = cls(
            embed_dim=embed_dim,
            num_heads=native_attn.num_heads,
            qkv_bias=qkv.bias is not None,
            attn_drop=attn_drop_p,
            proj_drop=proj_drop_p,
            q_type=q_type,
            quantization_bias=quantization_bias,
        )

        decomposed.attn_dim = qkv.out_features // 3
        decomposed.head_dim = decomposed.attn_dim // decomposed.num_heads
        if hasattr(native_attn, "scale"):
            decomposed.scale = float(native_attn.scale)
        # Force the explicit attention math path for decomposed timm attention so
        # downstream FX replacement can quantize q*scale and the attention matmuls.
        decomposed.fused_attn = False

        # Rebuild proj to match native attention dims exactly.
        decomposed.proj = nn.Linear(
            in_features=proj.in_features,
            out_features=proj.out_features,
            bias=proj.bias is not None
        )

        decomposed.qkv.weight.data.copy_(qkv.weight.data)
        if qkv.bias is not None and decomposed.qkv.bias is not None:
            decomposed.qkv.bias.data.copy_(qkv.bias.data)

        decomposed.proj.weight.data.copy_(proj.weight.data)
        if proj.bias is not None and decomposed.proj.bias is not None:
            decomposed.proj.bias.data.copy_(proj.bias.data)

        if hasattr(native_attn, "q_norm"):
            decomposed.q_norm = copy.deepcopy(native_attn.q_norm)
        if hasattr(native_attn, "k_norm"):
            decomposed.k_norm = copy.deepcopy(native_attn.k_norm)
        if hasattr(native_attn, "norm"):
            decomposed.norm = copy.deepcopy(native_attn.norm)

        return decomposed


@OpRegistry.register("DecomposedMlpBlock")
class DecomposedMlpBlock(nn.Module, QuantizedLayerMixin):
    """
    Decomposed timm-style MLP block (fc1/fc2 + activation/dropout).
    Works with modules exposing at least: fc1, fc2.
    """
    def __init__(self, in_features, hidden_features, out_features=None, q_type="fp8_e4m3", quantization_bias=None):
        super().__init__()
        out_features = out_features if out_features is not None else in_features

        self.q_type = q_type
        self.quantization_bias = quantization_bias

        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = nn.GELU()
        self.drop1 = nn.Identity()
        self.norm = nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.drop2 = nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

    @classmethod
    def from_native(cls, native_mlp, q_type="fp8_e4m3", quantization_bias=None):
        if not (hasattr(native_mlp, "fc1") and hasattr(native_mlp, "fc2")):
            raise TypeError("Module does not look like a timm-style MLP block.")

        fc1 = native_mlp.fc1
        fc2 = native_mlp.fc2

        decomposed = cls(
            in_features=fc1.in_features,
            hidden_features=fc1.out_features,
            out_features=fc2.out_features,
            q_type=q_type,
            quantization_bias=quantization_bias,
        )

        decomposed.fc1.weight.data.copy_(fc1.weight.data)
        if fc1.bias is not None and decomposed.fc1.bias is not None:
            decomposed.fc1.bias.data.copy_(fc1.bias.data)

        decomposed.fc2.weight.data.copy_(fc2.weight.data)
        if fc2.bias is not None and decomposed.fc2.bias is not None:
            decomposed.fc2.bias.data.copy_(fc2.bias.data)

        if hasattr(native_mlp, "act"):
            decomposed.act = copy.deepcopy(native_mlp.act)
        if hasattr(native_mlp, "drop1"):
            decomposed.drop1 = copy.deepcopy(native_mlp.drop1)
        elif hasattr(native_mlp, "drop"):
            decomposed.drop1 = copy.deepcopy(native_mlp.drop)
        if hasattr(native_mlp, "norm"):
            decomposed.norm = copy.deepcopy(native_mlp.norm)
        if hasattr(native_mlp, "drop2"):
            decomposed.drop2 = copy.deepcopy(native_mlp.drop2)
        elif hasattr(native_mlp, "drop"):
            decomposed.drop2 = copy.deepcopy(native_mlp.drop)

        return decomposed

# Backward-compatible aliases (kept for external imports).
DecomposedTimmAttention = DecomposedQkvAttention
DecomposedTimmMlp = DecomposedMlpBlock
