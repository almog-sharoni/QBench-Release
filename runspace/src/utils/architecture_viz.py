import torch
import json
import itertools
import re
from torchview import draw_graph
import pydot

def clean_label(label_str):
    if not label_str: return "Unknown"
    # Extract from Torchview's HTML label format
    if "TABLE" in label_str:
        # Look for the first TD content before <BR/> or </TD>
        match = re.search(r'<TD[^>]*>(.*?)(?:<BR/>|</TD>)', label_str, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    # Fallback
    return label_str.strip('"<>\n\t ')

def generate_hierarchical_json(model, input_size, model_name="Root", depth=3):
    """
    Generate a JSON graph compatible with Cytoscape.js representing the model 
    architecture, supporting compound (nested) nodes.
    
    Instead of tracing low-level operations, this relies on torchview's module tracing.
    """
    model.eval()
    
    # We use depth to trace down to the primitive modules but retain hierarchy
    # We disable expand_nested_tensors if we just want module views.
    graph = draw_graph(model, input_size=input_size, 
                       device='meta' if hasattr(torch, 'device') else 'cpu',
                       expand_nested=True, depth=depth, hide_module_functions=True)
                       
    dot_str = graph.visual_graph.source
    
    # Parse dot string with pydot
    graphs = pydot.graph_from_dot_data(dot_str)
    if not graphs:
        raise ValueError("Could not parse generated DOT graph.")
    pydot_graph = graphs[0]
    
    elements = []
    
    def process_subgraph(sg, parent_id=None):
        sg_name = sg.get_name().strip('"')
        
        # In dict, pydot prefixes clusters with 'cluster_'
        node_id = sg_name
        label = sg_name.replace("cluster_", "")
        
        # Extract label from attributes if it exists
        if sg.get_label():
            parsed_label = clean_label(sg.get_label())
            if parsed_label != "Unknown": label = parsed_label
            
        is_cluster = sg_name.startswith("cluster_")
        
        if is_cluster:
            elements.append({
                "data": {
                    "id": node_id,
                    "label": label,
                    "parent": parent_id,
                    "type": "compound",
                    "color": "#e2e8f0"
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
                
            n_label = clean_label(node.get_label())
                
            # Formatting color. We can inject green if "Quantized" is in the label
            is_quantized = "Quantized" in n_label
            color = "#a7f3d0" if is_quantized else "#fde68a" # Green vs Yellow
            
            elements.append({
                "data": {
                    "id": n_name,
                    "parent": current_parent,
                    "label": n_label,
                    "type": "node",
                    "color": color
                }
            })
            
        for child_sg in sg.get_subgraphs():
            process_subgraph(child_sg, current_parent)

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
