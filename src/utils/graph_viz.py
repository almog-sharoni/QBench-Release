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
        
        for node in dot.get_nodes():
            node_name = node.get_name().strip('"') # pydot might add quotes
            
            # Find the corresponding FX node
            fx_node = None
            for n in traced.graph.nodes:
                if n.name == node_name:
                    fx_node = n
                    break
            
            if fx_node:
                color = None
                style = "filled"
                
                if fx_node.op == 'call_module':
                    module = traced.get_submodule(fx_node.target)
                    if isinstance(module, quantized_ops):
                        color = "#90EE90" # LightGreen
                        node.set_comment("Quantized Layer")
                    elif isinstance(module, supported_modules):
                        color = "#FFD700" # Gold
                        node.set_comment("Unquantized Supported Layer")
                    elif isinstance(module, (torch.nn.ReLU, torch.nn.Identity, torch.nn.Dropout)):
                         # Common structural layers
                         pass
                    else:
                        # Unsupported / Unknown
                        color = "#FFB6C1" # LightPink
                        node.set_comment("Unsupported Layer")
                        
                elif fx_node.op == 'call_function':
                    if fx_node.target in supported_functions:
                         color = "#FFD700" # Gold
                         pass
                
                if color:
                    node.set_fillcolor(color)
                    node.set_style(style)

        # Add Legend
        import pydot
        legend = pydot.Subgraph(graph_name="cluster_legend", label="Legend", rank="source", style="solid", color="black", fontsize="20")
        
        # Create nodes for legend
        node_quant = pydot.Node("legend_quant", label="Quantized Layer", shape="box", style="filled", fillcolor="#90EE90", fontsize="14")
        node_supp = pydot.Node("legend_supp", label="Unquantized (Supported)", shape="box", style="filled", fillcolor="#FFD700", fontsize="14")
        node_unsupp = pydot.Node("legend_unsupp", label="Unquantized (Unsupported)", shape="box", style="filled", fillcolor="#FFB6C1", fontsize="14") # LightPink
        node_other = pydot.Node("legend_other", label="Other / Untracked", shape="box", style="filled", fillcolor="white", fontsize="14")
        
        legend.add_node(node_quant)
        legend.add_node(node_supp)
        legend.add_node(node_unsupp)
        legend.add_node(node_other)
        
        # Force vertical layout with invisible edges
        legend.add_edge(pydot.Edge(node_quant, node_supp, style="invis"))
        legend.add_edge(pydot.Edge(node_supp, node_unsupp, style="invis"))
        legend.add_edge(pydot.Edge(node_unsupp, node_other, style="invis"))
        
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
