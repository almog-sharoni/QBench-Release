
import os

# Fix for container permission issues
os.environ['TORCH_HOME'] = '/tmp/torch'
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

import sys
import torch
import torch.nn as nn
import argparse
import csv
import numpy as np
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.src.adapters.adapter_factory import create_adapter
from runspace.src.registry.op_registry import OpRegistry

def get_args():
    parser = argparse.ArgumentParser(description="Analyze layer-wise bandwidth requirements (weights, inputs, outputs, MACs)")
    parser.add_argument("--model_name", type=str, default="resnet18", help="Model name")
    parser.add_argument("--weights", type=str, default="DEFAULT", help="Model weights")
    parser.add_argument("--dataset_name", type=str, default="imagenet", help="Dataset name")
    parser.add_argument("--dataset_path", type=str, default="/data/imagenet/val", help="Dataset path")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers (0 is recommended for single-batch analysis)")
    parser.add_argument("--output_dir", type=str, default="runspace/experiments/bandwidth_analysis", help="Output directory")
    parser.add_argument("--limit_batches", type=int, default=1, help="Number of batches to run (1 is usually enough for static shape analysis)")
    parser.add_argument("--exclude_types", type=str, default="BatchNorm2d,ReLU", help="Comma-separated list of layer types to exclude and fuse into previous layer (e.g., 'BatchNorm2d,ReLU')")
    parser.add_argument("--models_config", type=str, default=None, help="Path to a YAML file containing a list of models to run")
    return parser.parse_args()

class ResidualAdd(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        return x + y

def instrument_residuals(model):
    """
    Uses torch.fx to replace addition operations (likely residual connections)
    with a ResidualAdd module so we can hook it.
    """
    print("Instrumenting model with ResidualAdd modules...")
    try:
        traced = torch.fx.symbolic_trace(model)
        
        # Collect candidates first to avoid modifying graph while iterating
        candidates = []
        for node in traced.graph.nodes:
            if (node.op == 'call_function' and (node.target == torch.add or node.target == getattr(torch, "add", None) or node.target.__name__ == 'add')) or \
               (node.op == 'call_method' and node.target == 'add'):
                if len(node.args) >= 2:
                    candidates.append(node)
        
        # Modify graph
        count = 0
        for node in candidates:
             mod_name = f"residual_add_{count}"
             count += 1
             
             # Add module to the GraphModule instance
             traced.add_module(mod_name, ResidualAdd())
             
             # Replace node
             with traced.graph.inserting_after(node):
                 new_node = traced.graph.call_module(mod_name, args=node.args, kwargs=node.kwargs)
                 node.replace_all_uses_with(new_node)
        
        traced.recompile()
        return traced
        
    except Exception as e:
        print(f"Warning: Failed to instrument residuals: {e}")
        import traceback
        traceback.print_exc()
        return model

class BandwidthTracker:
    def __init__(self):
        # We use a list to preserve exact execution order including duplicates if any (though get_hook prevents that per layer)
        self.stats_list = []
        self.recorded_layers = set()
        
    def get_hook(self, layer_name, layer):
        def hook(module, input, output):
            # Only process once per layer
            if layer_name in self.recorded_layers:
                return

            # input is a tuple (x, y, ...), output is tensor
            # For ResidualAdd, input has 2 tensors.
            total_input_elements = 0
            input_shapes = []
            
            if isinstance(input, tuple):
                for x in input:
                    if isinstance(x, torch.Tensor):
                        total_input_elements += x.numel()
                        input_shapes.append(str(tuple(x.shape)))
                        # Assume batch size from first tensor
                        if 'batch_size' not in locals():
                            batch_size = x.shape[0]
            elif isinstance(input, torch.Tensor):
                 total_input_elements = input.numel()
                 input_shapes.append(str(tuple(input.shape)))
                 batch_size = input.shape[0]
            else:
                 batch_size = 0
            
            num_inputs = total_input_elements
            input_shape_str = str(input_shapes) if len(input_shapes) > 1 else (input_shapes[0] if input_shapes else "N/A")

            if isinstance(output, torch.Tensor):
                 num_outputs = output.numel()
                 output_shape = str(tuple(output.shape))
            else:
                 num_outputs = 0
                 output_shape = "N/A"
            
            # Weights
            num_weights = 0
            if hasattr(module, 'weight') and module.weight is not None:
                num_weights = module.weight.numel()
            if hasattr(module, 'bias') and module.bias is not None:
                num_weights += module.bias.numel()

            # MACs Calculation
            macs = 0
            macs_status = "ok" # ok, zero_implied, unknown_zero
            
            if isinstance(module, nn.Linear):
                in_features = module.in_features
                macs = num_outputs * in_features
                
            elif isinstance(module, nn.Conv2d):
                k_h, k_w = module.kernel_size
                in_c = module.in_channels
                groups = module.groups
                ops_per_output = (k_h * k_w * in_c) // groups
                macs = num_outputs * ops_per_output
                macs = num_outputs * ops_per_output
            
            elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm2d)):
                 # Normalization: approx 5-10 ops per element (mean, var, sub, div, scale, shift)
                 # For BW analysis, keeping it simple: 1 Op per element (or just treat as 0 low compute).
                 # User wants to avoid "Unknown Op" warning.
                 macs = num_outputs
                 macs_status = "ok"

            elif isinstance(module, nn.MultiheadAttention):
                 # MHA: Q, K, V projections + Attention
                 # Simplified Estimate assuming Q, K, V are linear projections of input dim E
                 # Total MACs roughly 4 * Num_Outputs * Embed_Dim
                 # 3 Projections (3 * E^2 * L) + Out Proj (1 * E^2 * L) = 4 * E^2 * L
                 # Num_Outputs = B * L * E.
                 # So MACs = 4 * Num_Outputs * E
                 embed_dim = module.embed_dim
                 if embed_dim > 0:
                     macs = 4 * num_outputs * embed_dim
                 macs_status = "ok"

            elif isinstance(module, ResidualAdd):
                # Element-wise addition: 1 Op per output element
                macs = num_outputs 
                macs_status = "ok"

            elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
                # Pooling: kernel_size elements per output.
                k = module.kernel_size
                if isinstance(k, int):
                    k_h, k_w = k, k
                else:
                    k_h, k_w = k
                
                macs = num_outputs * k_h * k_w
                macs_status = "ok"
            
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                # AdaptiveAvgPool2d: Often used for global average pooling.
                # Ops approx equal to Num_Inputs (accumulation of all inputs to outputs).
                macs = num_inputs
                macs_status = "ok"

            elif isinstance(module, (nn.BatchNorm2d, nn.ReLU, nn.Dropout, nn.Identity)):
                # Known to have 0 MACs
                macs = 0
                macs_status = "zero_known"
            
            else:
                # Unknown operation
                if num_weights > 0:
                     macs_status = "unknown_weights"
                else:
                     macs_status = "unknown_zero"

            record = {
                'layer_name': layer_name,
                'layer_type': module.__class__.__name__,
                'num_weights': num_weights,
                'input_shape': input_shape_str,
                'output_shape': output_shape,
                'num_inputs': num_inputs,
                'num_outputs': num_outputs,
                'num_macs': macs,
                'macs_status': macs_status,
                'batch_size': batch_size
            }
            
            self.stats_list.append(record)
            self.recorded_layers.add(layer_name)
            
        return hook

def verify_coverage(model, captured_layers):
    """
    Checks if there are any call_module nodes in the FX graph 
    that were NOT captured by our hooks.
    """
    print("\n--- Verifying Layer Coverage with Torch.FX ---")
    try:
        # Symbolic Trace
        # We need to handle cases where model has custom trace issues, but for standard models ok.
        # Note: If we already instrumented the model, 'model' here IS the GraphModule.
        if isinstance(model, torch.fx.GraphModule):
            graph_module = model
        else:
            graph_module = torch.fx.symbolic_trace(model)
        
        missed_modules = []
        
        for node in graph_module.graph.nodes:
            if node.op == 'call_module':
                target_name = node.target
                # Check if this target was recorded
                if target_name not in captured_layers:
                    # Sometimes targets are weird or submodules we didn't hook? 
                    # We hooked all leaf modules. FX 'call_module' targets usually are leaf modules or containers.
                    # If we missed it, print it.
                    missed_modules.append(target_name)
        
        if missed_modules:
            print(f"Warning: The following modules appear in FX trace but were not hooked (or not executed?):")
            for m in missed_modules[:20]:
                print(f" - {m}")
            if len(missed_modules) > 20:
                print(f" ... and {len(missed_modules)-20} more.")
        else:
            print("Success! All FX call_module nodes were captured.")
            
    except Exception as e:
        print(f"FX Verification failed (non-critical): {e}")

def process_single_model(model_name, weights, dataset_config, args, device):
    """
    Process analysis for a single model.
    """
    # Output Dir
    output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine Resolution based on model name
    # Heuristics for common models
    base_size = 224
    resize = 256
    
    if "inception" in model_name:
        base_size = 299
        resize = 299 # Inception usually uses 299 directly or slightly larger? 
                     # Torchvision reference uses Resize(299), CenterCrop(299) for Inception usually?
                     # Actually standard is Resize(299) then CenterCrop(299) or Resize(342) Crop(299)?
                     # Weights.transforms() says: Resize(342), CenterCrop(299).
        resize = 342
        
    elif "efficientnet" in model_name:
        # Simple mapping for b0-b7
        if "b0" in model_name: base_size = 224; resize=256
        elif "b1" in model_name: base_size = 240; resize=256 # approx
        elif "b2" in model_name: base_size = 260; resize=288
        elif "b3" in model_name: base_size = 300; resize=320
        elif "b4" in model_name: base_size = 380; resize=384
        elif "b5" in model_name: base_size = 456; resize=456
        elif "b6" in model_name: base_size = 528; resize=528
        elif "b7" in model_name: base_size = 600; resize=600
        # V2 variants
        elif "v2_s" in model_name: base_size = 384; resize=384 # V2-S is optim for 384? Checks needed. V2-S weights are 384.
        elif "v2_m" in model_name: base_size = 480; resize=480
        elif "v2_l" in model_name: base_size = 480; resize=480
    
    dataset_config['image_size'] = base_size
    dataset_config['resize_size'] = resize
    
    # Load Model
    print(f"Loading model {model_name} (Resolution: {base_size}x{base_size})...")
    config = {
        'model': {'name': model_name, 'weights': weights},
        'adapter': {'type': 'generic', 'quantized_ops': []}, 
        'dataset': dataset_config
    }
    
    try:
        adapter = create_adapter(config)
        model = adapter.model
        # Instrument Residuals BEFORE moving to device
        model = instrument_residuals(model)
        
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to load or instrument model {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return

    # Data Loader
    from runspace.core.runner import Runner
    runner = Runner(device)
    loader = None
    try:
        loader = runner.setup_data_loader(config)
    except Exception as e:
        print(f"Warning: Could not setup data loader ({e}). Trying dummy input.")

    # Register Hooks
    tracker = BandwidthTracker()
    handles = []
    
    print("Registering hooks on all leaf modules...")
    for name, module in model.named_modules():
        # Hook if leaf (no children) OR if it is MultiheadAttention (which might have internal helpers but we want to treat as one op context)
        # Note: If we hook MHA, we should ideally skip its children to avoid double-counting if they are also looped.
        # But named_modules includes children.
        # However, BandwidthTracker.get_hook checks 'recorded_layers'. We should ensure we don't record children if parent is MHA.
        # Better strategy: explicitly check if parent is MHA and skip.
        # But parent info is not easy in named_modules.
        
        # Simplified: Just hook MHA. If children are hooked later, they will be separate entries.
        # BandwidthTracker uses `recorded_layers` to avoid dups of SAME layer name.
        # It does not prevent child hooking.
        # But 'MultiheadAttention' usually encapsulates linear layers that are also named_modules.
        # If we hook both, we get both entries.
        
        should_hook = len(list(module.children())) == 0 or isinstance(module, nn.MultiheadAttention)
        if should_hook:
            handles.append(module.register_forward_hook(tracker.get_hook(name, module)))
            
    # Run Inference
    print("Running inference...")
    with torch.no_grad():
        if loader:
            for i, batch in enumerate(loader):
                if i >= args.limit_batches: break
                images, labels = adapter.prepare_batch(batch)
                images = images.to(device)
                model(images)
        else:
            sz = dataset_config.get('image_size', 224)
            print(f"Using dummy input [B, 3, {sz}, {sz}]")
            dummy = torch.randn(args.batch_size, 3, sz, sz).to(device)
            model(dummy)

    # Remove hooks
    for h in handles: h.remove()
    
    # Post-Process: Fusion
    raw_stats = tracker.stats_list
    fused_stats = []
    
    # Parse exclude types
    exclude_list = [t.strip() for t in args.exclude_types.split(',') if t.strip()]

    # Auto-exclude activations from OpRegistry
    from runspace.src.registry.op_registry import OpRegistry
    supported_ops = OpRegistry.get_supported_ops()
    for orig_cls, quant_cls in supported_ops.items():
        if OpRegistry.is_activation(quant_cls.__name__):
            if orig_cls.__name__ not in exclude_list:
                exclude_list.append(orig_cls.__name__)

    if exclude_list:
        print(f"Applying fusion logic (excluding: {exclude_list})...")
        current_layer = None
        
        for record in raw_stats:
            l_type = record['layer_type']
            
            if current_layer is None:
                record['num_weights'] = str(record['num_weights'])
                current_layer = record
                
            elif l_type in exclude_list:
                current_layer['layer_name'] += f" + {record['layer_name']}"
                current_layer['layer_type'] += f" + {l_type}"
                current_layer['num_weights'] += f" + {record['num_weights']}"
                current_layer['num_macs'] += record['num_macs']
                current_layer['output_shape'] = record['output_shape']
                current_layer['num_outputs'] = record['num_outputs']
            else:
                fused_stats.append(current_layer)
                record['num_weights'] = str(record['num_weights'])
                current_layer = record
        
        if current_layer:
            fused_stats.append(current_layer)
            
    else:
        fused_stats = raw_stats
        for r in fused_stats: r['num_weights'] = str(r['num_weights'])

    # Calculate Bandwidth Metrics (128 ALUs)
    import math
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed. Skipping plots.")
        plt = None
        
    for layer in fused_stats:
        macs = layer['num_macs']
        # Ensure macs is int
        if isinstance(macs, str):
             # Handle fused string "X + Y" -> this shouldn't happen for MACs as we summed them?
             # wait, in fusion logic: current_layer['num_macs'] += record['num_macs'] (int sum)
             pass
             
        cycles = math.ceil(macs / 128) if macs > 0 else 0
        
        # Calculate Input Data (Weights + Inputs)
        # Note: num_weights might be a string expression "X + Y" if fused.
        # We need to evaluate it or sum it. 
        # In fusion logic: current_layer['num_weights'] += f" + {record['num_weights']}"
        # We should probably sum the values for bandwidth calc.
        
        # Helper to parse "A + B" string or int
        def parse_val(v):
            if isinstance(v, (int, float)): return v
            if isinstance(v, str):
                return sum(float(x) for x in v.split('+') if x.strip())
            return 0
            
        w_count = parse_val(layer['num_weights'])
        i_count = parse_val(layer['num_inputs'])
        o_count = parse_val(layer['num_outputs'])
        
        # Calculate per-component BW (Elements/Cycle)
        if cycles > 0:
            bw_weights = w_count / cycles
            bw_inputs = i_count / cycles
            bw_outputs = o_count / cycles
        else:
            bw_weights = 0
            bw_inputs = 0
            bw_outputs = 0
            
        layer['cycles'] = cycles
        layer['input_data_elements'] = i_count + w_count # Keep for Data Vol plot
        layer['output_data_elements'] = o_count
        
        layer['bw_weights_epc'] = bw_weights
        layer['bw_inputs_epc'] = bw_inputs
        layer['bw_outputs_epc'] = bw_outputs
        layer['bw_total_elements_per_cycle'] = bw_weights + bw_inputs + bw_outputs

    # Visualization
    if plt:
        layer_types = [l['layer_type'] for l in fused_stats]
        short_names = [n[:20] + "..." if len(n) > 20 else n for n in layer_types]
        
        # Plot 1: Data Volume (Data Volume plot remains mostly same or we could separate weights/inputs there too?)
        # User asked for B/W separation specifically. Let's separate it in Data Volume too for consistency?
        # Current Data Volume: Inputs+Weights vs Outputs. Labels: 'Inputs + Weights', 'Outputs'
        # Let's keep Data Volume simple or separate if easy. Let's separate.
        
        w_vols = [parse_val(l['num_weights']) for l in fused_stats]
        i_vols = [parse_val(l['num_inputs']) for l in fused_stats]
        o_vols = [parse_val(l['num_outputs']) for l in fused_stats]
        
        plt.figure(figsize=(max(10, len(layer_types)*0.3), 6))
        x = np.arange(len(layer_types))
        width = 0.35
        
        # Stack Inputs + Weights?
        plt.bar(x - width/2, i_vols, width, label='Inputs', color='skyblue')
        plt.bar(x - width/2, w_vols, width, bottom=i_vols, label='Weights', color='orange')
        plt.bar(x + width/2, o_vols, width, label='Outputs', color='lightgreen')
        
        plt.xlabel('Layers')
        plt.ylabel('Elements (Count) - Log Scale')
        plt.yscale('log')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.title(f'{model_name} Data Volume per Layer')
        plt.xticks(x, short_names, rotation=90, fontsize=8)
        
        # Highlight uncertain layers
        def highlight_labels(ax, stats):
            labels = ax.get_xticklabels()
            for i, label in enumerate(labels):
                if stats[i].get('macs_status') == 'unknown_weights':
                    label.set_color('red')
                    label.set_fontweight('bold')
                    
        highlight_labels(plt.gca(), fused_stats)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'data_volume.png'))
        plt.close()
        

        # Plot 2: Bandwidth Requirements (Bytes/Cycle) - 3 Subplots (FP32, FP8, FP4)
        # Separate Inputs, Weights, Outputs as grouped bars.
        
        fig, axes = plt.subplots(3, 1, figsize=(max(12, len(layer_types)*0.4), 12), sharex=True)
        
        precisions = [('FP32', 4), ('FP8', 1), ('FP4', 0.5)]
        
        # Component Data (Elements/Cycle)
        # Ensure we use numpy arrays
        bw_w = np.array([l['bw_weights_epc'] for l in fused_stats])
        bw_i = np.array([l['bw_inputs_epc'] for l in fused_stats])
        bw_o = np.array([l['bw_outputs_epc'] for l in fused_stats])
        
        colors = {'Inputs': 'skyblue', 'Weights': 'orange', 'Outputs': 'lightgreen'}
        bar_width = 0.25 # split 0.8 space into 3
        
        # We need groups of 3 bars per layer
        for i, (prec_name, multiplier) in enumerate(precisions):
            ax = axes[i]
            
            # Calculate Bytes/Cycle
            b_w = bw_w * multiplier
            b_i = bw_i * multiplier
            b_o = bw_o * multiplier
            
            # Plot Bars
            # Input: x - bar_width
            # Weight: x
            # Output: x + bar_width
            
            ax.bar(x - bar_width, b_i, bar_width, label='Inputs', color=colors['Inputs'])
            ax.bar(x, b_w, bar_width, label='Weights', color=colors['Weights'])
            ax.bar(x + bar_width, b_o, bar_width, label='Outputs', color=colors['Outputs'])
            
            ax.set_ylabel(f'{prec_name} BW (B/C)\n(Log Scale)')
            ax.set_yscale('log')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Annotate Highest Value in this subplot
            local_max_val = -1
            local_max_pos = (0, 0)
            
            # Check all bars in this subplot
            for data, offset in [(b_i, -bar_width), (b_w, 0), (b_o, bar_width)]:
                if len(data) > 0:
                    current_max = np.max(data)
                    if current_max > local_max_val:
                        local_max_val = current_max
                        idx = np.argmax(data)
                        local_max_pos = (x[idx] + offset, current_max)
            
            if local_max_val > 0:
                 ax.annotate(f'{local_max_val:.1f}', 
                             xy=local_max_pos, 
                             xytext=(0, 5), textcoords='offset points',
                             ha='center', va='bottom',
                             fontsize=8, weight='bold')

        # X-Axis Labels (Bottom only)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(short_names, rotation=90, fontsize=8)
        axes[2].set_xlabel('Layers')
        highlight_labels(axes[2], fused_stats) 
        
        # Legend (inputs, weights, outputs) - Put on top plot
        # Use simple handles
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors['Inputs'], label='Inputs'),
            Patch(facecolor=colors['Weights'], label='Weights'),
            Patch(facecolor=colors['Outputs'], label='Outputs')
        ]
        axes[0].legend(handles=legend_elements, loc='upper right')
        
        axes[0].set_title(f'{model_name} Required Bandwidth (Grouped Components)')
        
        # Formula Text (Bottom of Figure)
        formula_text = r"$BW_{type} = \frac{Elements_{type}}{\lceil MACs / 128 \rceil}$"
        fig.text(0.5, 0.01, f"Formula: {formula_text}", ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.8, "pad":5})

        
        # Annotate Highest Value
        # Calculate totals for annotation
        bw_total_epc = bw_w + bw_i + bw_o
        all_bars = [
            (bw_total_epc * 4, x - bar_width, 'FP32'),
            (bw_total_epc * 1, x, 'FP8'),
            (bw_total_epc * 0.5, x + bar_width, 'FP4')
        ]
        
        global_max_val = -1
        global_max_pos = (0, 0)
        
        for data, x_pos, label in all_bars:
            # data is numpy array here from stacked bars calc above? 
            # bw_w, bw_i, bw_o were np arrays. bw_total_epc is np array.
            
            max_val = np.max(data)
            if max_val > global_max_val:
                global_max_val = max_val
                # Find index
                idx = np.argmax(data)
                global_max_pos = (x_pos[idx], max_val)

        # Draw annotation
                global_max_pos = (x_pos[idx], max_val)

        # Draw annotation
        plt.annotate(f'{global_max_val:.2f} B/C', 
                     xy=global_max_pos, 
                     xytext=(0, 10), textcoords='offset points',
                     ha='center', va='bottom',
                     fontsize=9, weight='bold',
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        plt.legend()
        plt.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust for figtext
        plt.savefig(os.path.join(output_dir, 'bandwidth_requirements.png'))
        plt.close()

        # Plot 3: Runtime Bottleneck Analysis (10 Bytes/Cycle) using Stacked Bar Chart
        # --------------------------------------------------------------------------------
        
        TARGET_BW = 1.0 # Bytes/Cycle
        BYTES_PER_ELEM = 1.0 # Assuming INT8/FP8 for bandwidth constraints analysis
        
        # Calculate Times
        t_inputs = []
        t_weights = []
        t_outputs = []
        t_computes = []
        layer_labels = []
        label_colors = []
        
        for layer in fused_stats:
             # Calculate Data Transfer Cycles
             # T = (Elements * Bytes/Elem) / BW
             w_count = parse_val(layer['num_weights'])
             i_count = parse_val(layer['num_inputs'])
             o_count = parse_val(layer['num_outputs'])
             
             t_in = (i_count * BYTES_PER_ELEM) / TARGET_BW
             t_w = (w_count * BYTES_PER_ELEM) / TARGET_BW
             t_out = (o_count * BYTES_PER_ELEM) / TARGET_BW
             t_total_data = t_in + t_w + t_out
             
             # Compute Cycles (Already calculated as ceil(MACs / 128))
             t_comp = layer['cycles']
             
             t_inputs.append(t_in)
             t_weights.append(t_w)
             t_outputs.append(t_out)
             t_computes.append(t_comp)
             layer_labels.append(layer['layer_type'])
             
             # Classification using simple rule: Limit is max(Compute, Data)
             # But usually "Compute Limited" means Compute Time > Data Time
             # "Bandwidth Limited" means Data Time > Compute Time
             if t_total_data > t_comp:
                 label_colors.append('red') # B/W Limited
             else:
                 label_colors.append('green') # Compute Limited

        # Plotting
        plt.figure(figsize=(max(10, len(layer_types)*0.3), 8))
        x = np.arange(len(layer_types))
        width = 0.6
        
        # We want to show the COMPOSITION of the runtime
        # If Compute Limited: The bar height is T_compute. But we also want to show underlying data time?
        # User asked: "bars for inputs, weights, outputs run time separateley" 
        # This implies stacking them?
        # AND "mark layers in two colors" - this refers to the layer names on Y axis or X axis labels.
        # Let's stack Inputs + Weights + Outputs to show "Data Time".
        # And plot "Compute Time" as a separate bar or overlay?
        # A common way for bottleneck: 
        # Bar 1: Stack(In, W, Out) -> Total Data Time
        # Bar 2: Compute Time 
        # But this doubles the bars.
        # User asked for "bars for inputs, weights, outputs run time" -> implies stacking these 3.
        # And then classification based on sum(these 3) vs Compute.
        # Let's verify if user implies Compute should also be a bar? "bars for inputs ,weights outputs run time (separatley)"
        # Maybe: 
        # Y-Axis: Layer Names.
        # X-Axis: Time (Cycles).
        # Bar: Stacked (Input, Weight, Output). 
        # But where is compute? "run time" usually is max(Data, Compute).
        # If we only stack Input/Weight/Output, we show Data Time.
        # If we want to verify coverage, we typically overlay compute or show it side-by-side.
        # Given constraint "bars for inputs, weights, outputs", I will stack these.
        # And maybe add a marker for Compute Time? Or a separate bar?
        # Let's do a grouped bar? 
        # Group 1: Stacked (In, W, Out)
        # Group 2: Compute
        # This clearly shows the bottleneck.
        
        plt.figure(figsize=(12, max(8, len(layer_types)*0.4))) # Taller for horizontal bars
        
        y_pos = np.arange(len(layer_labels))
        height = 0.4
        
        # Data Transfer Stack
        p1 = plt.barh(y_pos + height/2, t_inputs, height, label='Input Time', color='skyblue')
        p2 = plt.barh(y_pos + height/2, t_weights, height, left=t_inputs, label='Weight Time', color='orange')
        # correct 'left' for 3rd bar is sum of previous two
        left_for_out = [i+w for i,w in zip(t_inputs, t_weights)]
        p3 = plt.barh(y_pos + height/2, t_outputs, height, left=left_for_out, label='Output Time', color='lightgreen')
        
        # Compute Bar (Side by side)
        p4 = plt.barh(y_pos - height/2, t_computes, height, label='Compute Time', color='gray', alpha=0.7)
        
        plt.xlabel(f'Duration (Cycles) @ {TARGET_BW} B/Cycle (1 B/Elem)')
        plt.ylabel('Layers')
        plt.title(f'{model_name} Runtime Bottleneck Analysis\n(Red=B/W Limited, Green=Compute Limited)')
        
        plt.yticks(y_pos, [l[:30] for l in layer_labels], fontsize=9)
        
        # Color the Y-axis labels
        ax = plt.gca()
        y_labels = ax.get_yticklabels()
        for i, label in enumerate(y_labels):
            label.set_color(label_colors[i])
            label.set_fontweight('bold')
            
        plt.legend()
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.draw()
        
        # Invert Y axis to have first layer at top
        plt.gca().invert_yaxis()
        
        plt.savefig(os.path.join(output_dir, 'runtime_bottleneck.png'))
        plt.close()

        # --------------------------------------------------------------------------------
        # New Analysis: Bandwidth Sensitivity Sweep (0.1 - 3.0 B/C)
        # --------------------------------------------------------------------------------
        print("Running Bandwidth Sensitivity Sweep...")
        
        bw_points = np.arange(0.1, 3.1, 0.1) # 0.1 to 3.0
        total_cycles_curve = []
        
        # We also want to find the exact cycles at BW=1.0 for the point on the graph
        cycles_at_1_0 = 0
        
        for bw in bw_points:
            model_total_cycles = 0
            for layer in fused_stats:
                # 1. Compute Cycles
                t_comp = layer['cycles']
                
                # 2. Data Limit Cycles @ bw
                # Data = Inputs + Weights + Outputs (Elements)
                # Assuming 1 Byte/Element per user request/code context (or use BYTES_PER_ELEM=1.0)
                # If we want to accept different precisions, we should stick to a standard. 
                # User said "b/w = 1/10 bytes/cycle to 3 bytes/cycle". This implies Bytes is the unit.
                # And usually we assume Bytes per element is fixed for the analysis (e.g. 1 Byte/Elem for INT8/FP8)
                
                total_data_bytes = (parse_val(layer['num_inputs']) + parse_val(layer['num_weights']) + parse_val(layer['num_outputs'])) * 1.0 
                
                t_data = total_data_bytes / bw
                
                # Layer Time = max(Compute, Data)
                model_total_cycles += max(t_comp, t_data)
            
            total_cycles_curve.append(model_total_cycles)
            if np.isclose(bw, 1.0):
                cycles_at_1_0 = model_total_cycles

        # Plot 4: Bandwidth Sensitivity
        plt.figure(figsize=(10, 6))
        plt.plot(bw_points, total_cycles_curve, marker='o', linestyle='-', color='purple', label='Total Execution Time')
        
        # Mark BW=1.0
        idx_1_0 = np.argmin(np.abs(bw_points - 1.0))
        val_1_0 = total_cycles_curve[idx_1_0]
        plt.plot(1.0, val_1_0, marker='*', markersize=15, color='red', label='Roof (1.0 B/C)')
        
        plt.xlabel('Bandwidth (Bytes/Cycle)')
        plt.ylabel('Total Estimated Cycles')
        plt.title(f'{model_name} Performance vs Bandwidth')
        plt.grid(True, which="both", ls="--", alpha=0.6)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'bandwidth_sensitivity.png'))
        plt.close()

        # --------------------------------------------------------------------------------
        # New Analysis: Roofline Model & Histogram
        # --------------------------------------------------------------------------------
        # Roofline Plot: 
        # X: Arithmetic Intensity (Ops/Byte)
        # Y: Performance (Ops/Cycle)
        # Roof: Min(Peak_Compute, Intensity * BW)
        # We assume Peak Compute = 128 Ops/Cycle (from code: macs/128)
        # We assume Peak BW = 1.0 Byte/Cycle (from user req "roof should be 1 byte/cycle")
        
        PEAK_COMPUTE = 128.0
        PEAK_BW = 1.0
        
        intensities = []
        performances = []
        layer_names_roof = []
        required_bws = []
        
        for layer in fused_stats:
            macs = layer['num_macs']
            total_data_bytes = (parse_val(layer['num_inputs']) + parse_val(layer['num_weights']) + parse_val(layer['num_outputs'])) * 1.0
            
            if total_data_bytes > 0:
                intensity = macs / total_data_bytes
            else:
                intensity = 0 # Should not happen usually
            
            # Attainable Performance at Peak BW
            # Time = max(MACs/Peak_Compute, Bytes/Peak_BW)
            # Perf = MACs / Time
            time_cycles = max(macs / PEAK_COMPUTE, total_data_bytes / PEAK_BW)
            if time_cycles > 0:
                perf = macs / time_cycles
            else:
                perf = 0
            
            # Required BW to be Compute Bound
            # MACs/Compute_Peak = Bytes / Req_BW  -> Req_BW = Bytes * Compute_Peak / MACs = Compute_Peak / Intensity
            if intensity > 0:
                req_bw = PEAK_COMPUTE / intensity
            else:
                req_bw = 0
                
            intensities.append(intensity)
            performances.append(perf)
            layer_names_roof.append(layer['layer_type'])
            required_bws.append(req_bw)

        # Plot 5: Roofline Scatter (Classic Memory Wall Style - Linear Scale)
        plt.figure(figsize=(10, 6))
        
        # Determine Ranges (Linear)
        if intensities:
            max_intensity = max(intensities)
            max_perf = max(performances)
        else:
            max_intensity = 10.0
            max_perf = PEAK_COMPUTE

        # X Range: 0 to max observed intensity + buffer
        # Ensure we show the knee (PEAK / BW)
        knee_x = PEAK_COMPUTE / PEAK_BW
        x_limit = max(max_intensity * 1.2, knee_x * 1.5) 
        
        x_range = np.linspace(0, x_limit, 200)
        
        # 1. Compute Bound Line (Horizontal)
        y_compute = np.full_like(x_range, PEAK_COMPUTE)
        
        # 2. Memory Bound Line (Diagonal): Y = X * BW
        y_memory = x_range * PEAK_BW
        
        # Roof is min of both
        y_roof = np.minimum(y_compute, y_memory)
        
        plt.plot(x_range, y_roof, 'k-', linewidth=2, label=f'Roofline (BW={PEAK_BW} B/C)')
        
        # Scatter Layers
        plt.scatter(intensities, performances, c='blue', alpha=0.6, edgecolors='k', s=50, label='Layers')
        
        # Annotations
        plt.text(x_limit * 0.5, PEAK_COMPUTE * 1.05, 'Compute Roof', fontsize=10, fontweight='bold', color='gray')
        
        # Memory Wall text (along the diagonal)
        # Position at half the knee
        mid_mem_x = knee_x * 0.5
        mid_mem_y = mid_mem_x * PEAK_BW
        if mid_mem_x < x_limit:
            plt.text(mid_mem_x, mid_mem_y + PEAK_COMPUTE*0.05, 'Memory Wall', fontsize=10, fontweight='bold', color='gray', rotation=45)

        plt.xlabel('Arithmetic Intensity (Ops/Byte)')
        plt.ylabel('Performance (Ops/Cycle)')
        plt.title(f'{model_name} Roofline Analysis (Memory Wall Style - Linear, BW={PEAK_BW} B/C)')
        plt.grid(True, which="both", ls="--", alpha=0.6)
        plt.xlim(left=0, right=x_limit)
        plt.ylim(bottom=0, top=PEAK_COMPUTE * 1.3)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'roofline_plot.png'))
        plt.close()
        
        # Plot 6: Required Bandwidth Histogram
        # "roofline histogram for each model" - usually means histogram of arithmetic intensities or required BW.
        # Given "calc the points on it based on b/w", likely means distribution of bottlenecks.
        
        plt.figure(figsize=(10, 6))
        # Filter req_bw to reasonable range for histogram (avoid infinity)
        req_bws_filtered = [b for b in required_bws if b > 0 and b < 1000] # Cap outliers
        
        plt.hist(req_bws_filtered, bins=50, color='green', alpha=0.7, edgecolor='black')
        plt.axvline(x=PEAK_BW, color='red', linestyle='--', linewidth=2, label=f'Available BW ({PEAK_BW} B/C)')
        
        plt.xlabel('Required Bandwidth (Bytes/Cycle) to be Compute Bound')
        plt.ylabel('Number of Layers')
        plt.title(f'{model_name} Layer Bandwidth Requirements Histogram')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', alpha=0.5)
        
        # Add text for % layers bound
        num_mem_bound = sum(1 for b in req_bws_filtered if b > PEAK_BW)
        pct_mem_bound = (num_mem_bound / len(req_bws_filtered)) * 100 if req_bws_filtered else 0
        plt.text(0.95, 0.95, f"{pct_mem_bound:.1f}% Layers Memory Bound", 
                 transform=plt.gca().transAxes, ha='right', va='top', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                 
        plt.savefig(os.path.join(output_dir, 'required_bandwidth_histogram.png'), bbox_inches='tight')
        plt.close()

        # Plot 7: Layer PE Utilization vs Available Bandwidth
        # Requested: "byte/cycle vs utilatztion? (of 128 PEs)"
        # X: Available BW (Bytes/Cycle)
        # Y: PE Utilization % = min(100%, (BW * Intensity / PEAK_COMPUTE)*100)
        
        plt.figure(figsize=(12, 8))
        
        # Ranges
        num_points = 200
        bw_axis = np.linspace(0.1, 3.0, num_points)
        
        # Group color strategy
        # 1. Identify unique layer types
        unique_types = sorted(list(set(l['layer_type'] for l in fused_stats)))
        
        # 2. Assign Color Map
        import matplotlib.cm as cm
        # Use tab10 or tab20 for distinct categorical colors
        if len(unique_types) <= 10:
            cmap = cm.get_cmap('tab10')
        elif len(unique_types) <= 20:
            cmap = cm.get_cmap('tab20')
        else:
            cmap = cm.get_cmap('nipy_spectral') 
            
        type_to_color = {t: cmap(i/len(unique_types)) for i, t in enumerate(unique_types)}
        
        # Track plotted types for legend
        plotted_types = set()
        
        # Sort stats for drawing order (maybe by intensity so high intensity is on top? or bottom?)
        fused_stats.sort(key=lambda x: (parse_val(x['num_macs']) / ((parse_val(x['num_inputs']) + parse_val(x['num_weights']) + parse_val(x['num_outputs'])) * 1.0 + 1e-9)), reverse=True)

        for idx, layer in enumerate(fused_stats):
            macs = layer['num_macs']
            total_data_bytes = (parse_val(layer['num_inputs']) + parse_val(layer['num_weights']) + parse_val(layer['num_outputs'])) * 1.0
            
            if total_data_bytes > 0:
                intensity = macs / total_data_bytes
            else:
                intensity = 0 
            
            # Utilization = Performance / Peak
            y_util = np.minimum(1.0, (bw_axis * intensity) / PEAK_COMPUTE) * 100.0
            
            l_type = layer['layer_type']
            color = type_to_color[l_type]
            
            # Label only the first occurrence of each type for the legend
            label = l_type if l_type not in plotted_types else None
            if label:
                plotted_types.add(l_type)
            
            plt.plot(bw_axis, y_util, color=color, linewidth=1.5, alpha=0.8, label=label)
            
        plt.xlabel('Available Bandwidth (Bytes/Cycle)')
        plt.ylabel('PE Utilization (%)')
        plt.title(f'{model_name} Layer PE Utilization vs Bandwidth (128 PEs)')
        plt.grid(True, which="both", ls="--", alpha=0.6)
        plt.xlim(0.1, 3.0)
        plt.ylim(0, 105) # 0 to 100%
        
        # Reference Lines
        plt.axhline(y=100, color='red', linestyle='--', label='Max Utilization')
        plt.axvline(x=1.0, color='gray', linestyle='--', label='Current BW (1.0 B/C)')
        
        # Legend outside - categorized
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=1, title="Layer Types")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'utilization_sensitivity.png'), bbox_inches='tight')
        plt.close()

    # Verify Coverage
    verify_coverage(model, tracker.recorded_layers)

    # Check for Unknown MACs
    unknown_macs_layers = [r for r in raw_stats if r.get('macs_status') == 'unknown_weights']
    if unknown_macs_layers:
        print("\n[WARNING] The following layers have weights but 0 MACs were calculated (Unknown Op):")
        for r in unknown_macs_layers:
            print(f" - {r['layer_name']} ({r['layer_type']}): Weights={r['num_weights']}")
        print("Please review if these should have MACs implemented.\n")

    # Export
    csv_path = os.path.join(output_dir, "layer_bandwidth_stats.csv")
    print(f"Saving statistics to {csv_path}...")
    
    fieldnames = ['layer_type', 'num_weights', 'num_inputs', 'num_outputs', 'num_macs', 'cycles', 
                  'input_data_elements', 'output_data_elements', 
                  'bw_in_elements_per_cycle', 'bw_out_elements_per_cycle', 'bw_total_elements_per_cycle',
                  'macs_status', 'input_shape', 'output_shape', 'batch_size', 'layer_name']
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(fused_stats)
             
    print(f"Stats saved. Plots saved to {output_dir}")
    print(f"Done processing {model_name}.")


def main():
    args = get_args()
    
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset_config = {
         'name': args.dataset_name, 'path': args.dataset_path, 
         'batch_size': args.batch_size, 'num_workers': args.num_workers
    }
    
    models_to_run = []
    
    if args.models_config:
        import yaml
        print(f"Reading models configuration from {args.models_config}")
        with open(args.models_config, 'r') as f:
            models_config = yaml.safe_load(f)
            
        # Format expected: list of strings (model names) or list of dicts or dict with 'models' key
        # Assuming simple list of names as per user request context "models.yaml" usually has a list.
        # But let's support:
        # models:
        #   - resnet18
        #   - mobilenet_v2
        if isinstance(models_config, dict) and 'models' in models_config:
            raw_list = models_config['models']
        elif isinstance(models_config, list):
            raw_list = models_config
        else:
            print("Error: models_config must contain a list of models or a dict with 'models' key.")
            return

        for m in raw_list:
            if isinstance(m, str):
                models_to_run.append({'name': m, 'weights': 'DEFAULT'})
            elif isinstance(m, dict):
                models_to_run.append({'name': m.get('name'), 'weights': m.get('weights', 'DEFAULT')})
    else:
        models_to_run.append({'name': args.model_name, 'weights': args.weights})

    print(f"Found {len(models_to_run)} models to process.")
    
    for m_cfg in models_to_run:
        process_single_model(m_cfg['name'], m_cfg['weights'], dataset_config, args, device)

if __name__ == "__main__":
    main()

