import torch
import torch.nn as nn
import json
import itertools
import re
from torchview import draw_graph
import pydot

from ..ops.quant_arithmetic import QuantAdd, QuantMul
from ..ops.quant_matmul import QuantMatMul
from ..ops.quant_mha import (
    AttentionWeightedValues as RuntimeAttentionWeightedValues,
    DecomposedQkvAttention as RuntimeDecomposedQkvAttention,
    ScaledDotProduct as RuntimeScaledDotProduct,
)

metadata_registry = {}


class ViTEncoderBlockWrapper(nn.Module):
    """Replaces torchvision EncoderBlock for arch-graph visualization.

    Exposes the two residual skip-connections as explicit QuantAdd submodules
    so torchview can see them as named nodes instead of bare 'add' ops.
    """
    def __init__(self, block, q_type='fp8_e4m3', quantization_bias=None):
        super().__init__()
        for name, child in block.named_children():
            setattr(self, name, child)
        self.residual_add_1 = QuantAdd(q_type=q_type, quantization_bias=quantization_bias)
        self.residual_add_2 = QuantAdd(q_type=q_type, quantization_bias=quantization_bias)

    def forward(self, input):
        x = self.ln_1(input)
        attn_out, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(attn_out)
        x = self.residual_add_1(x, input)
        y = self.ln_2(x)
        y = self.mlp(y)
        return self.residual_add_2(x, y)


class TransposeLastTwo(nn.Module):
    def forward(self, x):
        return x.transpose(-2, -1)


class TransposeDims(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class ExpandKeyPaddingMask(nn.Module):
    def forward(self, key_padding_mask):
        return key_padding_mask.unsqueeze(1).unsqueeze(2)


class MaskedFillValue(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x, mask):
        return x.masked_fill(mask, self.value)


class MakeContiguous(nn.Module):
    def forward(self, x):
        return x.contiguous()


class ViewAttentionOutput(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, batch_size, tgt_len):
        return x.view(batch_size, tgt_len, self.embed_dim)


class ScaledDotProduct(nn.Module):
    def __init__(self, head_dim, q_type="fp8_e4m3", quantization_bias=None):
        super().__init__()
        self.head_dim = head_dim
        self.transpose_k = TransposeLastTwo()
        self.matmul = QuantMatMul(q_type=q_type, quantization_bias=quantization_bias)
        self.scale = QuantMul(q_type=q_type, quantization_bias=quantization_bias)
        self.add_mask = QuantAdd(q_type=q_type, quantization_bias=quantization_bias)
        self.expand_padding_mask = ExpandKeyPaddingMask()
        self.apply_padding_mask = MaskedFillValue(float('-inf'))
        self.register_buffer("scale_factor", torch.tensor(head_dim ** -0.5, dtype=torch.float32))

    def forward(self, q, k, attn_mask, key_padding_mask):
        k_t = self.transpose_k(k)
        scores = self.matmul(q, k_t)
        scores = self.scale(scores, self.scale_factor)

        if attn_mask is not None:
            scores = self.add_mask(scores, attn_mask)

        if key_padding_mask is not None:
            mask = self.expand_padding_mask(key_padding_mask)
            scores = self.apply_padding_mask(scores, mask)

        return scores


class AttentionWeightedValues(nn.Module):
    def __init__(self, embed_dim, q_type="fp8_e4m3", quantization_bias=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.matmul = QuantMatMul(q_type=q_type, quantization_bias=quantization_bias)
        self.transpose = TransposeDims(1, 2)
        self.make_contiguous = MakeContiguous()
        self.view_output = ViewAttentionOutput(embed_dim)

    def forward(self, attn_weights, v, batch_size, tgt_len):
        attn_output = self.matmul(attn_weights, v)
        attn_output = self.transpose(attn_output)
        attn_output = self.make_contiguous(attn_output)
        attn_output = self.view_output(attn_output, batch_size, tgt_len)
        return attn_output


class DecomposedQkvAttention(nn.Module):
    def __init__(self, runtime_attn, q_type="fp8_e4m3", quantization_bias=None):
        super().__init__()
        self.embed_dim = runtime_attn.embed_dim
        self.num_heads = runtime_attn.num_heads
        self.head_dim = runtime_attn.head_dim
        self.attn_dim = runtime_attn.attn_dim

        self.qkv = runtime_attn.qkv
        self.q_norm = runtime_attn.q_norm
        self.k_norm = runtime_attn.k_norm
        self.scaled_dot_product = ScaledDotProduct(
            self.head_dim,
            q_type=q_type,
            quantization_bias=quantization_bias,
        )
        self.softmax = runtime_attn.softmax
        self.attn_drop = runtime_attn.attn_drop
        self.attention_weighted_values = AttentionWeightedValues(
            self.attn_dim,
            q_type=q_type,
            quantization_bias=quantization_bias,
        )
        self.norm = runtime_attn.norm
        self.proj = runtime_attn.proj
        self.proj_drop = runtime_attn.proj_drop

    def forward(self, x, attn_mask=None, is_causal=False, **kwargs):
        del is_causal  # Graph-only wrapper follows the explicit unfused attention path.

        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)

        attn = self.scaled_dot_product(q, k, attn_mask, None)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        out = self.attention_weighted_values(attn, v, bsz, seq_len)
        out = self.norm(out)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


def _replace_attention_helpers_for_graph(model):
    restorations = []

    def visit(module):
        parent_q_type = getattr(module, 'q_type', 'fp8_e4m3')
        parent_quant_bias = getattr(module, 'quantization_bias', None)

        for name, child in list(module.named_children()):
            replacement = None

            if isinstance(child, RuntimeScaledDotProduct):
                replacement = ScaledDotProduct(
                    child.head_dim,
                    q_type=parent_q_type,
                    quantization_bias=parent_quant_bias,
                )
            elif isinstance(child, RuntimeAttentionWeightedValues):
                replacement = AttentionWeightedValues(
                    child.embed_dim,
                    q_type=parent_q_type,
                    quantization_bias=parent_quant_bias,
                )
            elif isinstance(child, RuntimeDecomposedQkvAttention):
                replacement = DecomposedQkvAttention(
                    child,
                    q_type=parent_q_type,
                    quantization_bias=parent_quant_bias,
                )
            elif (type(child).__name__ == 'EncoderBlock'
                  and 'torchvision.models' in type(child).__module__
                  and hasattr(child, 'self_attention')
                  and hasattr(child, 'mlp')):
                replacement = ViTEncoderBlockWrapper(
                    child,
                    q_type=parent_q_type,
                    quantization_bias=parent_quant_bias,
                )

            if replacement is not None:
                setattr(module, name, replacement)
                restorations.append((module, name, child))
                child = replacement

            visit(child)

    visit(model)
    return restorations

def clean_label(label_str):
    name, input_shape, output_shape, var_name, module_args = "Unknown", None, None, None, None
    if not label_str: return name, input_shape, output_shape, var_name, module_args
    
    # Extract var_name if available (from dynamic class hack)
    if "__VAR__" in label_str:
        # Expected format: __VAR__variable_name__VAR__ClassName
        parts = re.split(r'__VAR__', label_str)
        if len(parts) >= 3:
            v_part = parts[1]
            label_str = label_str.replace(f"__VAR__{v_part}__VAR__", "")
            if "__IDX__" in v_part:
                v_split = v_part.split("__IDX__")
                var_name = v_split[0]
                idx = int(v_split[1])
                if idx in metadata_registry:
                    module_args = metadata_registry[idx]
            else:
                var_name = v_part

    # Extract from Torchview's HTML label format
    if "TABLE" in label_str:
        # Look for the first TD content before <BR/> or </TD>
        match = re.search(r'<TD[^>]*>(.*?)(?:<BR/>|</TD>)', label_str, re.IGNORECASE | re.DOTALL)
        if match:
            name = match.group(1).strip()
            
        # extract input
        in_match = re.search(r'>input:<.*?<TD[^>]*>(.*?)</TD>', label_str, re.IGNORECASE | re.DOTALL)
        if in_match: input_shape = in_match.group(1).strip()
        
        # extract output
        out_match = re.search(r'>output:\s*<.*?<TD[^>]*>(.*?)</TD>', label_str, re.IGNORECASE | re.DOTALL)
        if out_match: output_shape = out_match.group(1).strip()
    else:
        # Fallback
        name = label_str.strip('"<>\n\t ')
        if name.startswith('cluster_'):
            name = name.replace('cluster_', '')
            
    return name, input_shape, output_shape, var_name, module_args

def generate_hierarchical_json(model, input_size, model_name="Root", depth=3):
    """
    Generate a JSON graph compatible with Cytoscape.js representing the model 
    architecture, supporting compound (nested) nodes.
    
    Instead of tracing low-level operations, this relies on torchview's module tracing.
    """
    model.eval()

    # --- Hack to expose variable names to TorchView ---
    restoration_list = []
    graph_module_restorations = _replace_attention_helpers_for_graph(model)
    global metadata_registry
    metadata_registry.clear()
    
    try:
        idx = 0
        for n, module in model.named_modules():
            if n:
                # get arguments using repr
                s = repr(module)
                args_str = ""
                # for simple leaf modules, get the inner arguments
                if '\n' not in s:
                    idx1 = s.find('(')
                    idx2 = s.rfind(')')
                    if idx1 != -1 and idx2 > idx1:
                        args_str = s[idx1+1:idx2]
                metadata_registry[idx] = args_str
                
                old_cls = module.__class__
                new_name = f"__VAR__{n}__IDX__{idx}__VAR__{old_cls.__name__}"
                idx += 1
                
                try:
                    new_cls = type(new_name, (old_cls,), {})
                    module.__class__ = new_cls
                    restoration_list.append((module, old_cls))
                except Exception:
                    pass
    except Exception as e:
        print("Warning: Variable name extraction failed", e)

    try:
        # Use CPU tracing for robustness.
        # Some models/custom modules can end up with meta tensors during torchview
        # tracing, which then breaks subsequent device moves.
        graph = draw_graph(
            model,
            input_size=input_size,
            device='cpu',
            expand_nested=True,
            depth=depth,
            hide_module_functions=True,
        )

        dot_str = graph.visual_graph.source
    finally:
        # Restore old class names
        for module, old_cls in restoration_list:
            module.__class__ = old_cls
        for parent, name, original_module in reversed(graph_module_restorations):
            setattr(parent, name, original_module)
    
    # Parse dot string with pydot
    graphs = pydot.graph_from_dot_data(dot_str)
    if not graphs:
        raise ValueError("Could not parse generated DOT graph.")
    pydot_graph = graphs[0]
    
    elements = []
    seen_labels = set()
    
    def process_subgraph(sg, parent_id=None):
        sg_name = sg.get_name().strip('"')
        
        # In dict, pydot prefixes clusters with 'cluster_'
        node_id = sg_name
        label = sg_name.replace("cluster_", "")
        
        # Extract label from attributes if it exists
        if sg.get_label():
            n, i_s, o_s, var_name, module_args = clean_label(sg.get_label())
            if n != "Unknown": label = n
            
        is_cluster = sg_name.startswith("cluster_")
        
        if is_cluster:
            is_first = label not in seen_labels
            seen_labels.add(label)
            
            elements.append({
                "data": {
                    "id": node_id,
                    "label": label,
                    "parent": parent_id,
                    "type": "compound",
                    "color": "#e2e8f0",
                    "var_name": var_name,
                    "module_args": module_args,
                    "is_first_appearance": is_first
                }
            })
            current_parent = node_id
        else:
            current_parent = parent_id
            
        for node in sg.get_nodes():
            n_name = node.get_name().strip('"')
            
            # Skip hidden nodes or graphviz internal nodes
            if n_name in ("node", "edge", "graph") or n_name.startswith("struct"):
                continue
                
            n_label, i_s, o_s, var_name, module_args = clean_label(node.get_label())
                
            # Formatting color. We explicitly check for Quant in the class name itself!
            is_quantized = "Quant" in n_label or "quant" in (var_name or "").lower()
            color = "#a7f3d0" if is_quantized else "#fde68a" # Green vs Yellow
            
            elements.append({
                "data": {
                    "id": n_name,
                    "parent": current_parent,
                    "label": n_label,
                    "type": "node",
                    "color": color,
                    "var_name": var_name,
                    "input_shape": i_s,
                    "output_shape": o_s,
                    "module_args": module_args
                }
            })
            
        # Group child subgraphs
        child_sgs = sg.get_subgraphs()
        grouped_children = []
        current_group = []
        current_label = None
        
        for c_sg in child_sgs:
            raw_lbl = c_sg.get_label()
            if not raw_lbl: 
                lbl = c_sg.get_name()
            else:
                lbl, _, _, _, _ = clean_label(raw_lbl)
                
            if lbl == current_label:
                current_group.append(c_sg)
            else:
                if current_group:
                    grouped_children.append((current_label, current_group))
                current_group = [c_sg]
                current_label = lbl
                
        if current_group:
            grouped_children.append((current_label, current_group))
            
        for label, group in grouped_children:
            if len(group) >= 3 and label and "cluster_" not in label:
                # Create a compound node for this repetitive group
                # Using 3 or more repeating structures to trigger collapsing (e.g. 4x BasicBlock)
                group_id = f"{current_parent}_group_{id(group)}"
                is_first = f"group_{label}" not in seen_labels
                seen_labels.add(f"group_{label}")
                
                elements.append({
                    "data": {
                        "id": group_id,
                        "label": f"{len(group)}x {label}",
                        "parent": current_parent,
                        "type": "compound",
                        "color": "#cbd5e1",
                        "is_group": True,
                        "is_first_appearance": is_first
                    }
                })
                for c_sg in group:
                    process_subgraph(c_sg, group_id)
            else:
                for c_sg in group:
                    process_subgraph(c_sg, current_parent)

    # Process root
    process_subgraph(pydot_graph)
    
    # Process edges
    edge_idx = 0
    for edge in pydot_graph.get_edges():
        src = edge.get_source().strip('"')
        dst = edge.get_destination().strip('"')
        elements.append({
            "data": {
                "id": f"e{edge_idx}",
                "source": src,
                "target": dst
            }
        })
        edge_idx += 1
        
    return json.dumps(elements)

if __name__ == "__main__":
    from torchvision import models
    model = models.resnet18()
    js = generate_hierarchical_json(model, (1, 3, 224, 224), depth=3)
    print(js[:800] + "...")
