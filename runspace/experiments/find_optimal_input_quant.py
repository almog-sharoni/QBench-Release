
import os
import sys
import torch
import torch.nn as nn
import argparse
import numpy as np
import yaml
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.src.adapters.adapter_factory import create_adapter
from runspace.src.registry.op_registry import OpRegistry
from runspace.core.runner import Runner
from runspace.src.quantization.quantizer import quantize
from runspace.src.ops.quant_base import quantize_tensor, calculate_scale
# from runspace.src.quantization.constants import get_quantization_bias

# Fix for container permission issues
os.environ['TORCH_HOME'] = '/tmp/torch'
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

baseline_formats = ['fp32', 'fp8_e4m3', 'fp8_e5m2','fp8_e3m4','fp8_e2m5','fp8_e1m6','fp8_e6m1','fp8_e7m0','fp8_e0m7']

candidate_formats = [
            # Signed FP8 (1 sign bit + 7 bits E/M)
            'fp8_e0m7','fp8_e1m6'   ,'fp8_e2m5','fp8_e3m4','fp8_e4m3','fp8_e5m2','fp8_e6m1','fp8_e7m0'
            # 'fp4_e3m0' , 'fp4_e2m1' , 'fp4_e1m2' , 'fp4_e0m3',
            
            # Unsigned FP8 (0 sign bit + 8 bits E/M) - Fully utilizes 8 bits
            # 'ufp4_e4m0', 'ufp4_e3m1', 'ufp4_e2m2', 'ufp4_e1m3', 'ufp4_e0m4'
            'ufp8_e8m0','ufp8_e7m1','ufp8_e6m2','ufp8_e5m3','ufp8_e4m4','ufp8_e3m5','ufp8_e2m6','ufp8_e1m7','ufp8_e0m8'

            # Expended FP8
            'efp8_e0m7','efp8_e1m6','efp8_e2m5','efp8_e3m4','efp8_e4m3','efp8_e5m2','efp8_e6m1','efp8_e7m0'
        ]
def get_args():
    parser = argparse.ArgumentParser(description="Find optimal input quantization (Dynamic)")
    parser.add_argument("--model_name", type=str, default="resnet18", help="Model name")
    parser.add_argument("--weights", type=str, default="DEFAULT", help="Model weights")
    parser.add_argument("--models_file", type=str, default=None, help="Path to models.yaml file to run on multiple models")
    parser.add_argument("--dataset_name", type=str, default="imagenet", help="Dataset name")
    parser.add_argument("--dataset_path", type=str, default="/data/imagenet/val", help="Dataset path")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers")
    parser.add_argument("--limit_batches", type=int, default=-1, help="Limit number of batches to process (default: -1 for all)")
    parser.add_argument("--output_dir", type=str, default="runspace/experiments/optimal_input_quant", help="Output directory")
    parser.add_argument("--metric", type=str, default="mse,l1", help="Comma-separated error metrics for dynamic selection (e.g. 'mse,l1')")
    parser.add_argument("--chunk_size", type=int, default=128, help="Chunk size for input quantization (blocks)")
    parser.add_argument("--only_dynamic", action="store_true", help="Skip baseline runs and only run dynamic optimization")
    # Add other args as needed
    return parser.parse_args()


class DynamicInputQuantizer:
    def __init__(self, model, metric='mse', chunk_size=128):
        self.model = model
        self.metric = metric
        self.chunk_size = chunk_size
        self.hooks = []
        self.layer_stats = {} # Store stats if needed
        self.running_error = 0.0 # Accumulate selected error (the optimization metric)
        self.total_chunks = 0
        
        # Comprehensive Error Tracking
        self.stats = {
            'sum_l1_err': 0.0,
            'sum_mse_err': 0.0,
            'sum_l1_norm': 0.0,
            'sum_l2_norm': 0.0
        }
        
        # Define candidate formats for dynamic selection
        self.candidate_formats = candidate_formats
        
        # Supported ops to hook
        self.supported_ops = tuple(OpRegistry.get_supported_ops().values())

    def register_hooks(self):
        """
        Register forward PRE-hooks to intercept and quantize inputs before they enter the layer.
        """
        count = 0
        for name, module in self.model.named_modules():
            # Hook into supported ops (Conv2d, Linear, etc.)
            if isinstance(module, self.supported_ops) or \
               isinstance(module, (nn.Conv2d, nn.Linear)):
                
                # Use partial to capture layer name
                hook = self._get_hook(name)
                self.hooks.append(module.register_forward_pre_hook(hook))
                count += 1
        print(f"Registered dynamic input quantization hooks on {count} layers.")
        print(f"Dynamic Metric: {self.metric.upper()}")

    def _get_hook(self, layer_name):
        def hook_fn(module, args):
            """
            Pre-hook: Receives input arguments (tuple). 
            Must return modified arguments (tuple) or None.
            """
            x = args[0] # Input tensor
            if not isinstance(x, torch.Tensor):
                return None # Skip if not tensor

            # ==========================================================
            # Dynamic Error Calculation & Selection (GPU)
            # ==========================================================
            
            # 1. Choose Format Dynamically & Quantize
            # _select_best_format now returns the quantized tensor directly
            # It also updates internal stats
            x_quantized = self._select_best_format(x, layer_name)
            
            # ==========================================================
            # 2. Track Comprehensive Stats (L1, MSE, Norms)
            # ==========================================================
            # Note: _select_best_format updates running_error for the *optimization* metric
            # But we want to track ALL metrics for the report.
            # We calculate this on the *final* quantized result.
            
            with torch.no_grad():
                diff = x - x_quantized
                diff_flat = diff.reshape(-1)
                x_flat = x.reshape(-1)
                
                l1_err = diff_flat.abs().sum().item()
                mse_err = diff_flat.pow(2).sum().item()
                
                l1_norm = x_flat.abs().sum().item()
                l2_norm = x_flat.pow(2).sum().item()
                
                self.stats['sum_l1_err'] += l1_err
                self.stats['sum_mse_err'] += mse_err
                self.stats['sum_l1_norm'] += l1_norm
                self.stats['sum_l2_norm'] += l2_norm
            
            # ==========================================================
            
            # Return new input tuple (replace x with x_quantized)
            if len(args) > 1:
                return (x_quantized,) + args[1:]
            return (x_quantized,)
            
        return hook_fn

    def _select_best_format(self, tensor, layer_name):
        """
        Analyze tensor and return the QUANTIZED tensor using the best format PER CHUNK.
        """
        return self._dynamic_quantize_per_chunk(tensor, layer_name)

    def _dynamic_quantize_per_chunk(self, tensor, layer_name):
        """
        Full GPU implementation of per-chunk dynamic format selection.
        1. Reshape to chunks.
        2. Simulate all formats -> calculate error per chunk.
        3. Select best format index per chunk.
        4. Gather best quantized chunks.
        5. Reconstruct tensor.
        """
        chunk_size = self.chunk_size
        device = tensor.device
        original_shape = tensor.shape
        
        # 1. Flatten and Pad to [TotalChunks, ChunkSize]
        if tensor.dim() > 1:
            flat = tensor.reshape(-1)
        else:
            flat = tensor
            
        num_elements = flat.numel()
        pad_len = 0
        if num_elements % chunk_size != 0:
            pad_len = chunk_size - (num_elements % chunk_size)
            flat = torch.nn.functional.pad(flat, (0, pad_len))
            
        num_chunks = flat.numel() // chunk_size
        # View as [NumChunks, ChunkSize]
        chunks = flat.view(num_chunks, chunk_size)
        
        # 2. Evaluate Candidates
        # Stack errors: [NumCandidates, NumChunks]
        # Store quantized versions: [NumCandidates, NumChunks, ChunkSize]
        
        candidate_errors = []
        candidate_qs = []
        
        # Pre-calc to avoid repeated appends if possible, but list append is fast enough for <10 formats
        
        for fmt in self.candidate_formats:
            # Skip fp32 if present (we only want quantized)
            if fmt == 'fp32': 
                candidate_errors.append(torch.full((num_chunks,), float('inf'), device=tensor.device))
                candidate_qs.append(chunks) # Placeholder
                continue
                
            try:
                # Quantize Chunked (Manual to allow re-use of chunks tensor)
                # Calculate scale per chunk
                max_val = chunks.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
                # bias = get_quantization_bias(fmt)
                scale = calculate_scale(max_val, fmt)
                
                # Quantize
                # quantize() returns float32 simulated values
                q_chunks = quantize(chunks / scale, q_type=fmt, validate=False) * scale
                
                # Error
                diff = chunks - q_chunks
                
                if self.metric == 'mse':
                    # Sum of squared error per chunk (mean or sum doesn't change relative ordering)
                    # Using mean to be consistent with metric name
                    err = diff.pow(2).mean(dim=1) 
                elif self.metric == 'l1':
                    err = diff.abs().mean(dim=1)
                else:
                    err = diff.pow(2).mean(dim=1)
                    
                candidate_errors.append(err)
                candidate_qs.append(q_chunks)
                
            except Exception as e:
                # Fallback for failed formats
                candidate_errors.append(torch.full((num_chunks,), float('inf'), device=tensor.device))
                candidate_qs.append(chunks)

        # Stack
        # Errors: [NumCandidates, NumChunks]
        all_errors = torch.stack(candidate_errors, dim=0)
        
        # 3. Select Best Index per Chunk
        # [NumChunks] -> Indices in [0, NumCandidates-1]
        best_indices = torch.argmin(all_errors, dim=0)
        
        # 4. Gather Best Quantized Chunks
        # We need to gather from candidate_qs which is list of [NumChunks, ChunkSize]
        # Stack qs: [NumCandidates, NumChunks, ChunkSize]
        all_qs = torch.stack(candidate_qs, dim=0)
        
        # Expand indices for gather: [1, NumChunks, ChunkSize]
        # We want to select along dim 0.
        # indices shape: [NumChunks] -> [1, NumChunks, ChunkSize]
        gather_indices = best_indices.view(1, num_chunks, 1).expand(1, num_chunks, chunk_size)
        
        # Gather: Result [1, NumChunks, ChunkSize]
        best_qs = torch.gather(all_qs, 0, gather_indices).squeeze(0)
        
        # 5. Reconstruct
        flat_quantized = best_qs.view(-1)
        
        if pad_len > 0:
            flat_quantized = flat_quantized[:num_elements]
            
        quantized_tensor = flat_quantized.view(original_shape)
        
        # --- Update Stats (Counts per format) ---
        if layer_name not in self.layer_stats:
            self.layer_stats[layer_name] = {'format_counts': {}}
            
        counts_dict = self.layer_stats[layer_name]['format_counts']
        
        # bincount on GPU is fast
        counts = torch.bincount(best_indices, minlength=len(self.candidate_formats))
        counts_cpu = counts.cpu().tolist()
        
        for idx, count in enumerate(counts_cpu):
            if count > 0:
                fmt = self.candidate_formats[idx]
                counts_dict[fmt] = counts_dict.get(fmt, 0) + count
        
        # --- Update Error Stats (Optimization Metric) ---
        # Gather best errors: [NumChunks]
        best_errors = all_errors[best_indices, torch.arange(num_chunks, device=device)]
        self.running_error += best_errors.sum().item()
        self.total_chunks += num_chunks

        return quantized_tensor


    def get_selected_counts(self):
        """Return a dictionary of format counts per layer."""
        return self.layer_stats

    def get_final_stats(self):
        """Return computed normalized and total errors."""
        norm_l1 = self.stats['sum_l1_err'] / self.stats['sum_l1_norm'] if self.stats['sum_l1_norm'] > 0 else 0
        norm_mse = self.stats['sum_mse_err'] / self.stats['sum_l2_norm'] if self.stats['sum_l2_norm'] > 0 else 0
        return {
            'norm_l1': norm_l1,
            'norm_mse': norm_mse,
            'total_l1': self.stats['sum_l1_err'],
            'total_mse': self.stats['sum_mse_err']
        }

    def cleanup(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def run_baselines(args, device, formats):
    """
    Run multiple baseline evaluations in a single pass.
    Optimization: Loads model and data once, then for each batch, evaluates all formats.
    """
    print(f"\n--- Running Baselines (Optimized: {formats}) ---")
    
    # 1. Load Model (FP32 Weights)
    config = {
        'model': {'name': args.model_name, 'weights': args.weights},
        'adapter': {
            'type': 'generic', 
            'quantized_ops': [], # Ensure weights are FP32
            'input_quantization': False # We will manually quantize inputs
        },
        'dataset': {
             'name': args.dataset_name, 'path': args.dataset_path, 
             'batch_size': args.batch_size, 'num_workers': args.num_workers
        }
    }
    
    runner = Runner(device)
    adapter = create_adapter(config)
    model = adapter.model
    model.to(device)
    model.eval()
    
    loader = runner.setup_data_loader(config)
    
    # 2. Initialize Stats
    # Track: correct_top1, correct_top5, total, sum_l1_err, sum_mse_err, sum_l1_norm, sum_l2_norm
    results = {fmt: {
        'correct_top1': 0, 'correct_top5': 0, 'total': 0,
        'sum_l1_err': 0.0, 'sum_mse_err': 0.0, 
        'sum_l1_norm': 0.0, 'sum_l2_norm': 0.0
    } for fmt in formats}
    
    batch_count = 0
    limit = args.limit_batches if args.limit_batches > 0 else float('inf')
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Baselines"):
            if batch_count >= limit: break
            
            if adapter: images, labels = adapter.prepare_batch(batch)
            else: images, labels = batch
            
            images = images.to(device)
            labels = labels.to(device)
            
            # Pre-calculate norms for original image (for normalization)
            # Flatten once
            images_flat = images.reshape(-1)
            l1_norm = images_flat.abs().sum().item()
            l2_norm = images_flat.pow(2).sum().item() # Squared L2 norm
            
            # Evaluate each format
            for fmt in formats:
                x = images
                l1_err = 0.0
                mse_err = 0.0
                
                if fmt != 'fp32':
                    # Calculate Quantization Error
                    # Note: We need to validate=False for speed
                    # But for correct error calculation, we should try to match the 'dynamic' behavior
                    # The baselines use per-tensor (default) or whatever `quantize` does.
                    # Dynamic uses per-chunk. This is a fairness difference, but acceptable for "Baseline".
                    
                    try:
                        # For proper error tracking, we need to manually quantize and keep track
                        # Let's assume standard per-tensor quantization for baselines? 
                        # Or per-channel? Standard `quantize` is per-tensor if axis not set.
                        
                        max_val = x.abs().max()
                        # bias = get_quantization_bias(fmt)
                        scale = calculate_scale(max_val, fmt)
                        x_quant = quantize(x / scale, q_type=fmt, validate=False) * scale
                        
                        diff = x - x_quant
                        diff_flat = diff.reshape(-1)
                        l1_err = diff_flat.abs().sum().item()
                        mse_err = diff_flat.pow(2).sum().item() # Sum of squared error
                        
                        x = x_quant # Use quantized for inference
                    except Exception as e:
                        print(f"Quantization error for {fmt}: {e}")
                
                # Inference
                outputs = model(x)
                
                # Accuracy
                _, pred = outputs.topk(5, 1, True, True)
                pred = pred.t()
                correct = pred.eq(labels.view(1, -1).expand_as(pred))
                
                res_top1 = correct[:1].reshape(-1).float().sum(0, keepdim=True).item()
                res_top5 = correct[:5].reshape(-1).float().sum(0, keepdim=True).item()
                
                results[fmt]['total'] += labels.size(0)
                results[fmt]['correct_top1'] += res_top1
                results[fmt]['correct_top5'] += res_top5
                results[fmt]['sum_l1_err'] += l1_err
                results[fmt]['sum_mse_err'] += mse_err
                results[fmt]['sum_l1_norm'] += l1_norm
                results[fmt]['sum_l2_norm'] += l2_norm
            
            batch_count += 1
            
    # Calculate Final Metrics
    final_stats = {}
    for fmt in formats:
        total = results[fmt]['total']
        if total > 0:
            acc1 = 100. * results[fmt]['correct_top1'] / total
            acc5 = 100. * results[fmt]['correct_top5'] / total
            
            # Average Error (per element) would require total elements count
            # But Normalized is easier: Sum(Err) / Sum(Norm)
            # Norm L1 = Sum(|x-q|) / Sum(|x|)
            # Norm MSE = Sum((x-q)^2) / Sum(x^2)  (Relative Squared Error)
            
            norm_l1 = results[fmt]['sum_l1_err'] / results[fmt]['sum_l1_norm'] if results[fmt]['sum_l1_norm'] > 0 else 0
            norm_mse = results[fmt]['sum_mse_err'] / results[fmt]['sum_l2_norm'] if results[fmt]['sum_l2_norm'] > 0 else 0
            
            total_l1 = results[fmt]['sum_l1_err']
            total_mse = results[fmt]['sum_mse_err']
        else:
            acc1, acc5, norm_l1, norm_mse, total_l1, total_mse = 0, 0, 0, 0, 0, 0
            
        final_stats[fmt] = {
            'acc1': acc1, 'acc5': acc5, 
            'norm_l1': norm_l1, 'norm_mse': norm_mse,
            'total_l1': total_l1, 'total_mse': total_mse
        }
        
        print(f"Baseline {fmt}: Top1={acc1:.2f}%, Top5={acc5:.2f}%, NormL1={norm_l1:.4e}, NormMSE={norm_mse:.4e}")
        
    # Cleanup
    del model
    del adapter
    del loader
    gc.collect()
    torch.cuda.empty_cache()
        
    return final_stats

def plot_format_histogram(layer_stats, output_dir):
    """Generate histogram of selected formats."""
    import matplotlib.pyplot as plt
    
    print("Generating format distribution histogram...")
    
    # Aggregated counts across all layers
    total_counts = {}
    
    for layer, stats in layer_stats.items():
        if 'format_counts' in stats:
            for fmt, count in stats['format_counts'].items():
                total_counts[fmt] = total_counts.get(fmt, 0) + count
                
    if not total_counts:
        print("No format statistics found to plot.")
        return

    formats = list(total_counts.keys())
    counts = list(total_counts.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(formats, counts, color='skyblue')
    plt.xlabel('Format')
    plt.ylabel('Total Selections (Chunks)')
    plt.title('Dynamic Input Format Selection Distribution')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(output_dir, "format_distribution.png"))
    plt.close()
    print(f"Saved histogram to {os.path.join(output_dir, 'format_distribution.png')}")


def plot_layer_format_distribution(layer_stats, output_dir, metric):
    """Generate a stacked bar chart of format distribution per layer."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    print(f"Generating layer-wise format distribution for {metric}...")
    
    layers = list(layer_stats.keys())
    if not layers:
        print("No layer stats found.")
        return

    # Collect all unique formats
    all_formats = set()
    for stats in layer_stats.values():
        if 'format_counts' in stats:
            all_formats.update(stats['format_counts'].keys())
    
    sorted_formats = sorted(list(all_formats))
    if not sorted_formats:
        return

    # Prepare data: [n_layers, n_formats]
    data = np.zeros((len(layers), len(sorted_formats)))
    
    for i, layer in enumerate(layers):
        counts = layer_stats[layer].get('format_counts', {})
        for j, fmt in enumerate(sorted_formats):
            data[i, j] = counts.get(fmt, 0)
            
    # Stacked bars
    # Adjust figure size based on number of layers
    plt.figure(figsize=(max(12, len(layers)*0.3), 8))
    
    bottom = np.zeros(len(layers))
    # Colormap
    cmap = plt.get_cmap('tab20', len(sorted_formats))
    
    for j, fmt in enumerate(sorted_formats):
        plt.bar(layers, data[:, j], bottom=bottom, label=fmt, color=cmap(j))
        bottom += data[:, j]
        
    plt.xlabel('Layer')
    plt.ylabel('Count (Chunks)')
    plt.title(f'Layer-wise Format Distribution ({metric.upper()})')
    plt.xticks(rotation=90, fontsize=8)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3)
    
    save_path = os.path.join(output_dir, f"layer_format_distribution_{metric}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved layer distribution to {save_path}")


from runspace.experiments.utils.plotting import plot_accuracy_comparison


def process_single_model(args, model_config, device, metrics):
    """Process a single model: Baselines -> Dynamic Metrics -> Plots."""
    
    model_name = model_config['name']
    weights = model_config.get('weights', 'DEFAULT')
    
    print(f"\n###########################################################")
    print(f" PROCESSING MODEL: {model_name} (Weights: {weights})")
    print(f"###########################################################")
    
    # Model-specific output dir
    model_out_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(model_out_dir, exist_ok=True)
    
    all_results = []
    
    # --- 1. Run Baselines (Global) ---
    if not args.only_dynamic:
        
        # We need to constructing specific args/config for this model
        # Create a temp args-like object or just pass config dict? 
        # run_baselines uses 'args.model_name' etc. 
        # Let's modify run_baselines to accept model_config dict instead of just args, 
        # or temporarily patch args.
        
        # Patch args for this model (simplest refactor without changing everything signature)
        # Note: This modifies the passed args object, which is fine since we do it per loop
        args.model_name = model_name
        args.weights = weights
        
        baseline_stats = run_baselines(args, device, baseline_formats)
        
        for fmt, stats in baseline_stats.items():
            all_results.append({
                'output_name': f"Base_{fmt}", 
                'acc1': stats['acc1'],
                'acc5': stats['acc5'],
                'errors': {
                    'norm_l1': stats['norm_l1'],
                    'norm_mse': stats['norm_mse']
                }
            })
    else:
        print("\nSkipping Baselines (--only_dynamic set)")
        
    
    # --- 2. Run Dynamic Optimization Loop ---
    
    # Pre-load Model and Data ONCE
    print(f"\n[Optimization] Loading model and dataset once for all metrics...")
    
    config = {
        'model': {'name': model_name, 'weights': weights},
        'adapter': {'type': 'generic', 'quantized_ops': []}, 
        'dataset': {
             'name': args.dataset_name, 'path': args.dataset_path, 
             'batch_size': args.batch_size, 'num_workers': args.num_workers
        }
    }
    
    # Explicitly clean up before loading
    gc.collect()
    torch.cuda.empty_cache()

    try:
        adapter = create_adapter(config)
        model = adapter.model
        model.to(device)
        model.eval()
        
        runner = Runner(device)
        loader = runner.setup_data_loader(config)
        
        for metric in metrics:
            print(f"\n===========================================")
            print(f"Processing Metric: {metric.upper()} for {model_name}")
            print(f"===========================================")
            
            metric_out_dir = os.path.join(model_out_dir, metric)
            os.makedirs(metric_out_dir, exist_ok=True)
            
            # Initialize Quantizer
            quantizer_handler = DynamicInputQuantizer(model, metric=metric, chunk_size=args.chunk_size)
            quantizer_handler.register_hooks()
            
            # Track Accuracy
            correct_top1 = 0
            correct_top5 = 0
            total = 0
            batch_count = 0
            limit = args.limit_batches if args.limit_batches > 0 else float('inf')
            
            try:
                with torch.no_grad():
                    for batch in tqdm(loader, desc=f"Dynamic ({model_name}/{metric})"):
                        if batch_count >= limit: break
                        
                        if adapter is not None:
                             images, labels = adapter.prepare_batch(batch)
                        else:
                             images, labels = batch
                        
                        images = images.to(device)
                        labels = labels.to(device)
                        
                        outputs = model(images)
                        
                        # Top-1 and Top-5
                        _, pred = outputs.topk(5, 1, True, True)
                        pred = pred.t()
                        correct = pred.eq(labels.view(1, -1).expand_as(pred))
                        
                        res_top1 = correct[:1].reshape(-1).float().sum(0, keepdim=True).item()
                        res_top5 = correct[:5].reshape(-1).float().sum(0, keepdim=True).item()
                        
                        total += labels.size(0)
                        correct_top1 += res_top1
                        correct_top5 += res_top5
                        batch_count += 1
                
                acc1 = 100. * correct_top1 / total if total > 0 else 0
                acc5 = 100. * correct_top5 / total if total > 0 else 0
                
                final_stats = quantizer_handler.get_final_stats()
                
                print(f"\nDynamic Run ({metric.upper()}): Top1={acc1:.2f}%, Top5={acc5:.2f}%")
                print(f"Norm L1: {final_stats['norm_l1']:.4e}, Norm MSE: {final_stats['norm_mse']:.4e}")
                
                all_results.append({
                    'output_name': f"Dyn_{metric.upper()}", 
                    'acc1': acc1,
                    'acc5': acc5,
                    'errors': {
                        'norm_l1': final_stats['norm_l1'],
                        'norm_mse': final_stats['norm_mse']
                    }
                })
                
                plot_format_histogram(quantizer_handler.layer_stats, metric_out_dir)
                plot_layer_format_distribution(quantizer_handler.layer_stats, metric_out_dir, metric)
                
                import json
                stats_path = os.path.join(metric_out_dir, "layer_stats.json")
                with open(stats_path, 'w') as f:
                    # Save stats + accuracy
                    save_data = quantizer_handler.layer_stats
                    save_data['accuracy'] = {
                        'top1': acc1, 'top5': acc5,
                        'norm_l1': final_stats['norm_l1'],
                        'norm_mse': final_stats['norm_mse']
                    }
                    json.dump(save_data, f, indent=4)
            
            except KeyboardInterrupt:
                print("\nInterrupted.")
                quantizer_handler.cleanup() # Clean hooks before returning
                return 
            except Exception as e:
                print(f"Error processing {model_name} / {metric}: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Always cleanup hooks for this metric so the model is clean for the next one
                quantizer_handler.cleanup()
                print("Hooks removed.") 
                
    finally:
        # Clean up memory after ALL metrics are done for this model
        if 'model' in locals(): del model
        if 'adapter' in locals(): del adapter
        if 'runner' in locals(): del runner
        if 'loader' in locals(): del loader
        
        gc.collect()
        torch.cuda.empty_cache()

    # --- 3. Plot Final Comparison ---
    if all_results:
        plot_accuracy_comparison(all_results, model_out_dir)
        
        # Also save raw results to CSV/JSON for easy review
        import json
        with open(os.path.join(model_out_dir, "comparison_results.json"), 'w') as f:
             json.dump(all_results, f, indent=4)


def main():
    # Force generic cleanup at startup
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    args = get_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Parse metrics
    metrics = [m.strip().lower() for m in args.metric.split(',')]
    print(f"Metrics to process: {metrics}")
    
    # Determine models to run
    models_to_run = []
    
    if args.models_file:
        print(f"Loading models from: {args.models_file}")
        with open(args.models_file, 'r') as f:
            yaml_models = yaml.safe_load(f)
            # Ensure it's a list
            if isinstance(yaml_models, list):
                models_to_run = yaml_models
            else:
                print("Error: models.yaml must contain a list of models.")
                sys.exit(1)
    else:
        # Single model from args
        models_to_run = [{'name': args.model_name, 'weights': args.weights}]
        
    print(f"Found {len(models_to_run)} models to process.")
    
    # Initialize Output Dir
    os.makedirs(args.output_dir, exist_ok=True)

    for model_config in models_to_run:
        process_single_model(args, model_config, device, metrics)

if __name__ == "__main__":
    main()

