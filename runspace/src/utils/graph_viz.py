import torch
import torch.fx
from torch.fx.passes.graph_drawer import FxGraphDrawer
import os
import sys

# Ensure project root is in path for imports if running standalone
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.registry.op_registry import OpRegistry
from src.ops.quant_base import QuantizedLayerMixin
from src.ops.quant_mha import DecomposedMultiheadAttention

def generate_quantization_graph(model: torch.nn.Module, output_path: str, model_name: str = "model"):
    """
    Generates an SVG graph of the model using torch.fx, highlighting quantized layers.
    
    Args:
        model: The PyTorch model to visualize.
        output_path: Path to save the SVG file.
        model_name: Name of the model for the graph title.
    """
    try:
        # 1. Trace the model
        # We need a custom tracer to treat quantized layers as leaves
        class CoverageTracer(torch.fx.Tracer):
            def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
                # Treat DecomposedMHA as non-leaf to see internal Softmax/Linear
                if isinstance(m, DecomposedMultiheadAttention):
                    return False
                
                # Treat other Quantized Layers and Activations as leaves
                quantized_ops = tuple(OpRegistry.get_supported_ops().values())
                if isinstance(m, quantized_ops):
                    return True
                    
                return super().is_leaf_module(m, module_qualified_name)

        tracer = CoverageTracer()
        # We don't want to actually run the model, just trace it.
        # But some models might need dummy inputs or specific args to trace.
        # torch.fx.symbolic_trace might be safer if the model is simple.
        # However, CoverageTracer inherits from Tracer, so we use tracer.trace()
        
        # Note: Tracing might fail for complex dynamic control flow.
        # We'll wrap in try-except.
        graph = tracer.trace(model)
        traced = torch.fx.GraphModule(model, graph)
        
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
            for n in traced.graph.nodes:
                if n.name == node_name:
                    fx_node = n
                    break
            
            if fx_node:
                color = "#D3D3D3" # Default Gray (Structural)
                style = "filled"
                label = str(fx_node.op)
                
                if fx_node.op == 'call_module':
                    module = traced.get_submodule(fx_node.target)
                    label = type(module).__name__
                    if isinstance(module, quantized_ops):
                        color = "#90EE90" # LightGreen
                    elif isinstance(module, supported_modules):
                        color = "#FFD700" # Gold
                    elif isinstance(module, (torch.nn.ReLU, torch.nn.Identity, torch.nn.Dropout)):
                         color = "#D3D3D3" # Gray
                    else:
                        color = "#FFB6C1" # LightPink (Unsupported)
                        
                elif fx_node.op == 'call_function':
                    label = getattr(fx_node.target, '__name__', str(fx_node.target))
                    if fx_node.target in supported_functions:
                         color = "#FFD700" # Gold
                    else:
                         color = "#D3D3D3" # Gray
                
                elif fx_node.op == 'call_method':
                    label = str(fx_node.target)
                    color = "#D3D3D3" # Gray
                
                elif fx_node.op == 'get_attr':
                    label = "get_attr"
                    color = "#D3D3D3" # Gray
                
                elif fx_node.op == 'placeholder':
                    label = "Input"
                    color = "#D3D3D3" # Gray
                
                elif fx_node.op == 'output':
                    label = "Output"
                    color = "#D3D3D3" # Gray
                
                node.set_fillcolor(color)
                node.set_style(style)
                legend_items[label] = color

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
        
    except ImportError:
        print("Warning: graphviz or pydot not installed. Skipping graph generation.")
    except Exception as e:
        print(f"Warning: Failed to generate quantization graph: {e}")
        # import traceback
        # traceback.print_exc()
