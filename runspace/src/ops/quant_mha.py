import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..registry.op_registry import OpRegistry
from .quant_base import QuantizedLayerMixin

# not supported yet
@OpRegistry.register("DecomposedMultiheadAttention", original_cls=nn.MultiheadAttention) 
class DecomposedMultiheadAttention(nn.Module, QuantizedLayerMixin):
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

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None, average_attn_weights=True):
        # Assuming batch_first=True for simplicity as it's common in ViT
        # query, key, value shape: (batch_size, seq_len, embed_dim)
        
        batch_size, seq_len, _ = query.shape
        
        # Quantize inputs
        # Note: In self-attention, query, key, value are often the same tensor.
        # If they are the same object, quantizing one will effectively quantize all if done in-place,
        # but quantize_input returns a new tensor.
        # We should quantize each.
        
        query = self.quantize_input(query)
        key = self.quantize_input(key)
        value = self.quantize_input(value)
        
        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        # (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        # (batch_size, num_heads, seq_len, head_dim) @ (batch_size, num_heads, head_dim, seq_len) -> (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attn_mask is not None:
            scores = scores + attn_mask
            
        if key_padding_mask is not None:
             # key_padding_mask is usually (batch_size, seq_len), need to expand
             # scores is (batch_size, num_heads, seq_len, seq_len)
             # We mask the last dimension (keys)
             mask = key_padding_mask.unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, seq_len)
             scores = scores.masked_fill(mask, float('-inf'))

        attn_weights = self.softmax(scores)
        attn_weights = self.dropout_layer(attn_weights)
        
        # (batch_size, num_heads, seq_len, seq_len) @ (batch_size, num_heads, seq_len, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, v)
        
        # Combine heads
        # (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, num_heads, head_dim) -> (batch_size, seq_len, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
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
