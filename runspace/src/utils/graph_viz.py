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
from src.utils.fx_trace_utils import find_non_tensor_nodes, trace_quant_aware
from src.utils.model_input_utils import resolve_model_input_size


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
        # 1. Trace the model with the shared quant-aware tracer.
        _, _, traced = trace_quant_aware(model)

        # Static analysis: identify nodes that produce integers rather than
        # tensors (shape arithmetic like B*C*num_patch_h, rshift, lshift …).
        # We forward-propagate a "non-tensor" label through the graph starting
        # from known seeds, so we never need to run the model.
        _non_tensor_nodes = find_non_tensor_nodes(traced.graph)
        
        # 2. Draw the graph
        drawer = FxGraphDrawer(traced, model_name)
        dot = drawer.get_dot_graph()
        
        # 3. Style the nodes
        # Get registry info
        supported_modules = tuple(OpRegistry.get_supported_ops().keys())
        quantized_ops = tuple(OpRegistry.get_supported_ops().values())
        supported_functions = OpRegistry.get_supported_functions()
        
        legend_items = {} # Label -> Color

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
                    else:
                        color = "#FFB6C1" # LightPink (Unsupported)

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
        import pydot
        legend = pydot.Subgraph(graph_name="cluster_legend", label="Legend", rank="source", style="solid", color="black", fontsize="50",ranksep="0.1")
        
        # Sort legend items by color priority then name
        def get_sort_key(item):
            name, color = item
            priority = {
                "#90EE90": 0, # Green (Quantized)
                "#FFD700": 1, # Gold (Supported)
                "#FFB6C1": 2, # Pink (Unsupported)
                "#D3D3D3": 3  # Gray (Structural)
            }
            return (priority.get(color, 4), name)

        section_titles = {
            "#90EE90": "Quantized Layers",
            "#FFD700": "Supported Layers (Unquantized)",
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
    label_parts = []
    label_parts.append(f"name=%{name}")
    label_parts.append(f"op_code={op}")
    
    if op in ['call_module', 'call_function', 'call_method']:
        label_parts.append(target_name)
        
    if formatted_args:
        arg_str = ", ".join(formatted_args)
        # Use newlines if the label gets too long
        if len(arg_str) > 30:
            arg_str = ",\\n".join(formatted_args)
        label_parts.append(f"args=({arg_str})")
        
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
