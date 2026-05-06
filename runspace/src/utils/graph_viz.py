import torch
import torch.fx
from torch.fx.passes.graph_drawer import FxGraphDrawer
import os
import sys
import inspect

# Ensure project root is in path for imports if running standalone
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.registry.op_registry import OpRegistry
from src.utils.fx_trace_utils import find_non_tensor_nodes, trace_quant_aware, fx_trace_subtrees
from src.utils.model_input_utils import resolve_model_input_size


def _is_external_module(module: torch.nn.Module) -> bool:
    """A module treated as an FX leaf (registered in OpRegistry) but with
    internal submodules worth showing as a cluster — e.g. ObservedAttention
    contains attn_scores/attn_apply submodules that are useful to expose."""
    cls_name = type(module).__name__
    in_registry = cls_name in getattr(OpRegistry, '_registry', {})
    return in_registry and any(True for _ in module.children())


def _get_model_dummy_input(model: torch.nn.Module) -> torch.Tensor:
    """Return a dummy input tensor on the same device as the model."""
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device('cpu')

    _, c, h, w = resolve_model_input_size(model)
    return torch.zeros(1, c, h, w, device=device)


def generate_quantization_graph(model: torch.nn.Module, output_path: str, model_name: str = "model"):
    """
    Generates an SVG graph of the model using torch.fx, highlighting quantized layers.

    Args:
        model: The PyTorch model to visualize.
        output_path: Path to save the SVG file.
        model_name: Name of the model for the graph title.
    """
    try:
        # 1. Trace the model. Try whole-model first (cleanest output for
        # fully-traceable models). On failure, descend via fx_trace_subtrees
        # and render each child subtree as its own SVG — handles models
        # like Matching whose orchestration blocks a full trace but whose
        # children (SuperPoint, SuperGlue) trace fine.
        try:
            _, _, traced = trace_quant_aware(model)
            subtrees = [('', traced)]
        except Exception as whole_err:
            try:
                subtrees = list(fx_trace_subtrees(model))
                if not subtrees:
                    raise RuntimeError('fx_trace_subtrees yielded nothing')
                print(f"FX whole-model trace failed ({whole_err}); rendering {len(subtrees)} subtree(s) separately.")
            except Exception as trace_err:
                print(f"FX trace failed ({trace_err}); falling back to module-hierarchy graph.")
                _generate_hierarchy_graph(model, output_path, model_name)
                return

        # Multi-subtree: each subtree traces cleanly on its own, but the
        # whole model doesn't. Render each child as its own SVG by
        # recursively calling generate_quantization_graph on the child
        # module — its trace will succeed (we already verified above).
        if len(subtrees) > 1:
            base, ext = os.path.splitext(output_path)
            sub_paths = []
            for prefix, _ in subtrees:
                try:
                    child = model.get_submodule(prefix)
                except AttributeError:
                    continue
                safe = prefix.replace('.', '_') or 'root'
                sub_out = f"{base}.{safe}{ext}"
                generate_quantization_graph(child, sub_out, model_name=f"{model_name}.{prefix}" if prefix else model_name)
                sub_paths.append((prefix, sub_out))
            # Write an index file at the original output_path location so
            # callers that hardcode the path still get a useful pointer.
            index_path = output_path.replace('.svg', '.index.txt')
            with open(index_path, 'w') as f:
                f.write(f"Model {model_name!r} could not be FX-traced as a whole.\n")
                f.write(f"Per-subtree SVGs:\n")
                for prefix, p in sub_paths:
                    f.write(f"  - {prefix or '<root>'}: {os.path.basename(p)}\n")
            print(f"Multi-subtree index saved to {index_path}")
            return

        # Single-subtree (whole-model trace succeeded OR only one child traced).
        prefix, traced = subtrees[0]
        _non_tensor_nodes = find_non_tensor_nodes(traced.graph)

        # 2. Draw the graph
        drawer = FxGraphDrawer(traced, model_name)
        dot = drawer.get_dot_graph()

        # 3. Style the nodes
        import pydot
        supported_modules = tuple(OpRegistry.get_supported_ops().keys())
        quantized_ops = tuple(OpRegistry.get_supported_ops().values())
        supported_functions = OpRegistry.get_supported_functions()

        legend_items = {}  # Label -> Color

        for node in dot.get_nodes():
            node_name = node.get_name().strip('"') # pydot might add quotes
            
            # Find the corresponding FX node
            fx_node = None
            # Standard FxGraphDrawer uses the node name directly.
            # However, some pydot versions might prefix or escape names.
            for n in traced.graph.nodes:
                if n.name == node_name:
                    fx_node = n
                    break
            
            if fx_node:
                # Hide shape-arithmetic nodes (produce ints, not tensors).
                if fx_node.name in _non_tensor_nodes:
                    node.set_style('invis')
                    continue

                # Basic Styling based on Operation
                color = "#D3D3D3" # Default Gray (Structural)
                style = "filled"
                label_text = str(fx_node.op)

                if fx_node.op == 'call_module':
                    module = traced.get_submodule(fx_node.target)
                    label_text = type(module).__name__
                    if isinstance(module, quantized_ops):
                        color = "#90EE90" # LightGreen
                    elif isinstance(module, supported_modules):
                        color = "#FFD700" # Gold
                    elif isinstance(module, (torch.nn.ReLU, torch.nn.Identity, torch.nn.Dropout)):
                        color = "#D3D3D3" # Gray
                    elif _is_external_module(module):
                        # External leaf: determine color from its internal layers
                        internal_quant  = any(isinstance(c, quantized_ops)    for _, c in module.named_modules())
                        internal_supported = any(isinstance(c, supported_modules) for _, c in module.named_modules())
                        color = "#90EE90" if internal_quant else ("#FFD700" if internal_supported else "#D3D3D3")
                        # Expand internals as a DOT cluster subgraph
                        cluster = pydot.Subgraph(
                            graph_name=f'cluster_{node_name}',
                            label=f'{type(module).__name__} internals',
                            style='dashed', color='#888888', fontsize='10',
                        )
                        for child_name, child_mod in module.named_modules():
                            if not child_name or '.' in child_name:
                                continue  # only direct children
                            if isinstance(child_mod, quantized_ops):
                                c_color = "#90EE90"
                            elif isinstance(child_mod, supported_modules):
                                c_color = "#FFD700"
                            else:
                                continue  # skip non-tracked internals
                            c_id = f'cluster_{node_name}_{child_name}'
                            cluster.add_node(pydot.Node(
                                f'"{c_id}"',
                                label=f'{child_name}\\n({type(child_mod).__name__})',
                                shape='box', style='filled', fillcolor=c_color, fontsize='9',
                            ))
                            legend_items[type(child_mod).__name__] = c_color
                        dot.add_subgraph(cluster)
                    else:
                        from ..registry.op_registry import OpRegistry as _OR
                        _cs = _OR.get_compliance_status(type(module).__name__)
                        color = "#FFD580" if _cs == "FP32 required" else "#FFB6C1"

                elif fx_node.op == 'call_function':
                    label_text = getattr(fx_node.target, '__name__', str(fx_node.target))
                    if fx_node.target in supported_functions:
                         color = "#FFD700" # Gold
                    else:
                         color = "#D3D3D3" # Gray

                elif fx_node.op == 'call_method':
                    label_text = str(fx_node.target)
                    color = "#D3D3D3" # Gray

                elif fx_node.op == 'get_attr':
                    label_text = "get_attr"
                    color = "#D3D3D3" # Gray

                elif fx_node.op == 'placeholder':
                    label_text = "Input"
                    color = "#D3D3D3" # Gray

                elif fx_node.op == 'output':
                    label_text = "Output"
                    color = "#D3D3D3" # Gray

                # Construct a rich label for the node
                rich_label = _get_node_label(fx_node, traced)
                node.set_label(rich_label)
                node.set_fillcolor(color)
                node.set_style(style)
                # Use label_text for the legend key
                legend_items[label_text] = color

        # Hide edges that touch invisible (non-tensor) nodes
        for edge in dot.get_edges():
            src = edge.get_source().strip('"')
            dst = edge.get_destination().strip('"')
            if src in _non_tensor_nodes or dst in _non_tensor_nodes:
                edge.set_style('invis')

        # Add Legend
        legend = pydot.Subgraph(graph_name="cluster_legend", label="Legend", rank="source", style="solid", color="black", fontsize="50",ranksep="0.1")
        
        # Sort legend items by color priority then name
        def get_sort_key(item):
            name, color = item
            priority = {
                "#90EE90": 0, # Green (Quantized)
                "#FFD700": 1, # Gold (Supported)
                "#FFD580": 2, # Amber (FP32 required)
                "#FFB6C1": 3, # Pink (Unsupported)
                "#D3D3D3": 4  # Gray (Structural)
            }
            return (priority.get(color, 4), name)

        section_titles = {
            "#90EE90": "Quantized Layers",
            "#FFD700": "Supported Layers (Unquantized)",
            "#FFD580": "FP32 Required (coord/interp)",
            "#FFB6C1": "Unsupported Layers",
            "#D3D3D3": "Structural / Other"
        }

        sorted_items = sorted(legend_items.items(), key=get_sort_key)
        
        previous_node = None
        current_color_group = None
        
        for i, (name, color) in enumerate(sorted_items):
            # Check for new section
            if color != current_color_group:
                current_color_group = color
                title = section_titles.get(color, "Other")
                
                # Create section header node
                header_name = f"legend_header_{i}"
                header_node = pydot.Node(header_name, label=title, shape="plaintext", fontsize="40")
                legend.add_node(header_node)
                
                if previous_node:
                    legend.add_edge(pydot.Edge(previous_node, header_node, style="invis"))
                previous_node = header_node

            # Create legend node
            leg_node_name = f"legend_node_{i}"
            leg_node = pydot.Node(leg_node_name, label=name, shape="box", style="filled", fillcolor=color, fontsize="35")
            legend.add_node(leg_node)
            
            if previous_node:
                legend.add_edge(pydot.Edge(previous_node, leg_node, style="invis",minlen="0.2"))
            previous_node = leg_node
        
        dot.add_subgraph(legend)
        
        # 4. Save to SVG
        dot.write_svg(output_path)
        print(f"Quantization graph saved to {output_path}")
        
    except ImportError as e:
        raise ImportError("graphviz or pydot not installed; cannot generate graph") from e

def _get_node_label(fx_node: torch.fx.Node, traced: torch.fx.GraphModule) -> str:
    """
    Constructs a rich label for a torch.fx node, mapping positional arguments to parameter names.
    Returns a record-style label string for Graphviz.
    """
    name = fx_node.name
    op = fx_node.op
    target = fx_node.target
    args = fx_node.args
    kwargs = fx_node.kwargs
    num_users = len(fx_node.users)

    target_name = ""
    sig = None
    
    if op == 'placeholder':
        target_name = f"Input: {target}"
    elif op == 'output':
        target_name = "Output"
    elif op == 'get_attr':
        target_name = f"attr: {target}"
    elif op == 'call_module':
        module = traced.get_submodule(target)
        target_name = type(module).__name__
        try:
            sig = inspect.signature(module.forward)
        except:
            pass
    elif op == 'call_function':
        target_name = getattr(target, '__name__', str(target))
        # Handle aliases like _operator.add -> add
        if target_name.startswith('_operator.'):
            target_name = target_name.replace('_operator.', '')
            
        try:
            sig = inspect.signature(target)
            # If signature is (*args, **kwargs) or empty for some builtins, 
            # try to fallback to common mappings for known torch ops
            if not sig or 'args' in sig.parameters:
                sig = _guess_signature(target_name)
        except:
            sig = _guess_signature(target_name)
    elif op == 'call_method':
        target_name = str(target)
        sig = _guess_signature(target_name)
    else:
        target_name = str(op)

    formatted_args = []
    if sig:
        params = list(sig.parameters.values())
        # Handle self if it's there
        if params and params[0].name == 'self':
            params = params[1:]
            
        for i, arg in enumerate(args):
            if i < len(params):
                param_name = params[i].name
                arg_repr = f"%{arg.name}" if isinstance(arg, torch.fx.Node) else str(arg)
                formatted_args.append(f"{param_name}={arg_repr}")
            else:
                arg_repr = f"%{arg.name}" if isinstance(arg, torch.fx.Node) else str(arg)
                formatted_args.append(arg_repr)
        
        for k, v in kwargs.items():
            v_repr = f"%{v.name}" if isinstance(v, torch.fx.Node) else str(v)
            formatted_args.append(f"{k}={v_repr}")
    else:
        # Fallback for no signature (builtins, etc) 
        for arg in args:
             formatted_args.append(f"%{arg.name}" if isinstance(arg, torch.fx.Node) else str(arg))
        for k, v in kwargs.items():
             v_repr = f"%{v.name}" if isinstance(v, torch.fx.Node) else str(v)
             formatted_args.append(f"{k}={v_repr}")

    # Build structured record label
    # Format: { line1 | line2 | ... }
    # Using specific record separators | for fields
    def _esc(s: str) -> str:
        # Graphviz record labels treat {, }, |, < > as structural; any
        # such chars from arg values (e.g. dict-shaped output args) must
        # be escaped or they'll break the label parser.
        return (
            s.replace('\\', '\\\\')
             .replace('{', '\\{')
             .replace('}', '\\}')
             .replace('|', '\\|')
             .replace('<', '\\<')
             .replace('>', '\\>')
             .replace('"', '\\"')
        )

    label_parts = []
    label_parts.append(f"name=%{_esc(name)}")
    label_parts.append(f"op_code={_esc(op)}")

    if op in ['call_module', 'call_function', 'call_method']:
        label_parts.append(_esc(target_name))

    if formatted_args:
        arg_str = ", ".join(formatted_args)
        # Use newlines if the label gets too long
        if len(arg_str) > 30:
            arg_str = ",\\n".join(formatted_args)
        label_parts.append(f"args=({_esc(arg_str)})")

    label_parts.append(f"num_users={num_users}")
    
    # Return Graphviz record label
    return "{" + " | ".join(label_parts) + "}"

def _guess_signature(name: str):
    """
    Returns a mock signature object for common torch operations if inspect.signature fails.
    """
    import inspect
    
    # Common signatures for torch functions/methods
    SIG_MAPPING = {
        'conv2d': ['input', 'weight', 'bias', 'stride', 'padding', 'dilation', 'groups'],
        'linear': ['input', 'weight', 'bias'],
        'matmul': ['input', 'other'],
        'bmm': ['input', 'mat2'],
        'add': ['input', 'other', 'alpha'],
        'sub': ['input', 'other', 'alpha'],
        'mul': ['input', 'other'],
        'div': ['input', 'other'],
        'cat': ['tensors', 'dim'],
        'softmax': ['input', 'dim'],
        'relu': ['input', 'inplace'],
        'view': ['shape'],
        'reshape': ['shape'],
        'transpose': ['dim0', 'dim1'],
        'permute': ['dims'],
        'flatten': ['start_dim', 'end_dim'],
        'mean': ['dim', 'keepdim'],
        'sum': ['dim', 'keepdim'],
        'max_pool2d': ['input', 'kernel_size', 'stride', 'padding', 'dilation', 'ceil_mode'],
        'avg_pool2d': ['input', 'kernel_size', 'stride', 'padding', 'ceil_mode', 'count_include_pad'],
    }
    
    clean_name = name.split('.')[-1] # Handle torch.nn.functional.conv2d etc
    if clean_name in SIG_MAPPING:
        params = [
            inspect.Parameter(p, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for p in SIG_MAPPING[clean_name]
        ]
        return inspect.Signature(params)

    return None


def _generate_hierarchy_graph(model: torch.nn.Module, output_path: str, model_name: str):
    """
    Fallback renderer for models that cannot be traced by torch.fx (e.g. SuperPoint,
    SuperGlue with dynamic shape-based control flow). Walks named_modules() and
    renders a containment tree of leaf modules grouped by their dotted-path prefix.
    """
    import pydot
    try:
        from src.ops.quant_base import QuantizedLayerMixin
    except Exception:
        QuantizedLayerMixin = ()
    try:
        from src.ops.quant_mha import DecomposedMultiheadAttention
    except Exception:
        DecomposedMultiheadAttention = ()

    supported_modules = tuple(OpRegistry.get_supported_ops().keys())
    quantized_ops = tuple(OpRegistry.get_supported_ops().values())

    legend_items = {}  # Label -> Color

    def classify(module: torch.nn.Module):
        cls_name = type(module).__name__
        if isinstance(module, quantized_ops) or (
            QuantizedLayerMixin and isinstance(module, QuantizedLayerMixin)
        ):
            return cls_name, "#90EE90"  # Green: Quantized
        if isinstance(module, supported_modules):
            return cls_name, "#FFD700"  # Gold: Supported
        compliance = OpRegistry.get_compliance_status(cls_name)
        if compliance == "FP32 required":
            return cls_name, "#FFD580"  # Amber
        return cls_name, "#FFB6C1"      # Pink: Unsupported

    leaves = []  # list of (dotted_name, module)
    for name, module in model.named_modules():
        if not name:
            continue  # skip root
        is_leaf = len(list(module.children())) == 0 or (
            DecomposedMultiheadAttention and isinstance(module, DecomposedMultiheadAttention)
        )
        if not is_leaf:
            continue
        leaves.append((name, module))

    dot = pydot.Dot(graph_name=model_name, graph_type='digraph', rankdir='TB', compound='true')

    # Build a nested cluster structure from dotted paths.
    # Each intermediate path segment becomes a cluster; the final leaf is a node.
    clusters = {}  # path_prefix -> pydot.Subgraph

    def get_or_create_cluster(prefix_parts):
        if not prefix_parts:
            return dot
        key = '.'.join(prefix_parts)
        if key in clusters:
            return clusters[key]
        parent = get_or_create_cluster(prefix_parts[:-1])
        sub = pydot.Subgraph(
            graph_name=f'cluster_{key.replace(".", "_")}',
            label=prefix_parts[-1],
            style='dashed', color='#888888', fontsize='14',
        )
        clusters[key] = sub
        parent.add_subgraph(sub)
        return sub

    for name, module in leaves:
        parts = name.split('.')
        parent_cluster = get_or_create_cluster(parts[:-1])
        label_text, color = classify(module)
        legend_items[label_text] = color
        node_id = f'"{name}"'
        parent_cluster.add_node(pydot.Node(
            node_id,
            label=f'{parts[-1]}\\n({label_text})',
            shape='box', style='filled', fillcolor=color, fontsize='11',
        ))

    # Add Legend (same layout/priority as the FX path above).
    legend = pydot.Subgraph(
        graph_name="cluster_legend", label="Legend", rank="source",
        style="solid", color="black", fontsize="50", ranksep="0.1",
    )

    def get_sort_key(item):
        _, col = item
        priority = {
            "#90EE90": 0,
            "#FFD700": 1,
            "#FFD580": 2,
            "#FFB6C1": 3,
            "#D3D3D3": 4,
        }
        return (priority.get(col, 4), item[0])

    section_titles = {
        "#90EE90": "Quantized Layers",
        "#FFD700": "Supported Layers (Unquantized)",
        "#FFD580": "FP32 Required (coord/interp)",
        "#FFB6C1": "Unsupported Layers",
        "#D3D3D3": "Structural / Other",
    }

    previous_node = None
    current_color_group = None
    for i, (lbl, col) in enumerate(sorted(legend_items.items(), key=get_sort_key)):
        if col != current_color_group:
            current_color_group = col
            header = pydot.Node(
                f"legend_header_{i}", label=section_titles.get(col, "Other"),
                shape="plaintext", fontsize="40",
            )
            legend.add_node(header)
            if previous_node:
                legend.add_edge(pydot.Edge(previous_node, header, style="invis"))
            previous_node = header
        leg_node = pydot.Node(
            f"legend_node_{i}", label=lbl,
            shape="box", style="filled", fillcolor=col, fontsize="35",
        )
        legend.add_node(leg_node)
        if previous_node:
            legend.add_edge(pydot.Edge(previous_node, leg_node, style="invis", minlen="0.2"))
        previous_node = leg_node

    dot.add_subgraph(legend)
    dot.write_svg(output_path)
    print(f"Quantization graph saved to {output_path}")
