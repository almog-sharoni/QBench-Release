import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import copy
from tqdm import tqdm
from typing import Dict, List, Any, Tuple

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.core.runner import Runner
from runspace.core.config_factory import ConfigFactory
from src.adapters.adapter_factory import create_adapter
from src.registry.op_registry import OpRegistry
from src.ops.quant_base import QuantizedLayerMixin

class MetricCollector:
    def __init__(self, ref_model: nn.Module, quant_model: nn.Module, device: torch.device):
        self.ref_model = ref_model
        self.quant_model = quant_model
        self.device = device
        self.ref_activations = {}
        self.quant_activations = {}
        self.hooks = []
        
        # Storage for metrics
        # Structure: layer_name -> { 'ref': {stats}, 'int8': {stats}, 'fp8': {stats} }
        self.layer_metrics = {}

    def _get_ref_hook(self, name):
        def hook(module, input, output):
            # Capture output of the block/layer
            # For Conv/Linear, output is the activation
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            self.ref_activations[name] = out.detach()
        return hook

    def _get_quant_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            self.quant_activations[name] = out.detach()
        return hook

    def register_hooks(self):
        # We want to probe "block boundaries and heavy compute ops"
        # For simplicity, we'll probe all supported ops (Conv, Linear) and maybe some others if needed.
        # We match layers by name.
        
        supported_ops = tuple(OpRegistry.get_supported_ops().keys())
        quantized_ops = tuple(OpRegistry.get_supported_ops().values())

        # Register on Ref Model
        for name, module in self.ref_model.named_modules():
            if isinstance(module, supported_ops):
                self.hooks.append(module.register_forward_hook(self._get_ref_hook(name)))
        
        # Register on Quant Model
        for name, module in self.quant_model.named_modules():
            # Check if it's a quantized layer
            if isinstance(module, quantized_ops) or isinstance(module, QuantizedLayerMixin):
                self.hooks.append(module.register_forward_hook(self._get_quant_hook(name)))

    def clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def _compute_ref_stats(self, tensor: torch.Tensor) -> Dict[str, float]:
        # max_abs, rms, R, p99, p99.9, outlier_frac
        t_abs = tensor.abs()
        max_abs = t_abs.max().item()
        mean_sq = torch.mean(tensor.float() ** 2)
        rms = torch.sqrt(mean_sq).item()
        
        # Avoid div by zero
        R = max_abs / rms if rms > 1e-9 else 0.0
        
        # Percentiles (approximate if tensor is large, but exact is fine for batch)
        # Flatten for percentile
        t_flat = t_abs.flatten()
        k99 = int(0.99 * t_flat.numel())
        k999 = int(0.999 * t_flat.numel())
        
        # topk is faster than sort for high percentiles
        # but for 99%, sort might be better or quantile
        # torch.quantile requires float
        # Subsample if too large (limit to 1M elements for quantile)
        if t_flat.numel() > 1_000_000:
            # Random subsample
            indices = torch.randperm(t_flat.numel(), device=t_flat.device)[:1_000_000]
            t_sample = t_flat[indices]
        else:
            t_sample = t_flat

        p99 = torch.quantile(t_sample.float(), 0.99).item()
        p999 = torch.quantile(t_sample.float(), 0.999).item()
        
        # outlier_frac = mean(|x| > 8*rms)
        outlier_mask = t_abs > (8 * rms)
        outlier_frac = outlier_mask.float().mean().item()
        
        return {
            'max_abs': max_abs,
            'rms': rms,
            'R': R,
            'p99': p99,
            'p99.9': p999,
            'outlier_frac': outlier_frac
        }

    def _compute_compare_stats(self, ref: torch.Tensor, quant: torch.Tensor) -> Dict[str, float]:
        # Cosine similarity, SQNR, MSE
        # Ensure shapes match
        if ref.shape != quant.shape:
            # This might happen if quant layer changes shape (unlikely for Conv/Linear)
            return {}
            
        ref_f = ref.float().flatten()
        quant_f = quant.float().flatten()
        
        # Cosine
        cos_sim = torch.nn.functional.cosine_similarity(ref_f.unsqueeze(0), quant_f.unsqueeze(0)).item()
        
        # MSE
        mse = torch.nn.functional.mse_loss(ref_f, quant_f).item()
        
        # SQNR (dB) = 10 * log10( E[ref^2] / E[(ref-quant)^2] )
        # E[(ref-quant)^2] is MSE
        signal_power = torch.mean(ref_f ** 2).item()
        noise_power = mse
        
        if noise_power < 1e-12:
            sqnr = 100.0 # Cap at 100 dB
        else:
            sqnr = 10 * np.log10(signal_power / noise_power)
            
        return {
            'cosine': cos_sim,
            'mse': mse,
            'sqnr_db': sqnr
        }

    def collect_ref_metrics(self, data_loader, num_batches):
        self.ref_model.eval()
        self.ref_model.to(self.device)
        
        print(f"Collecting REF metrics on {num_batches} batches...")
        
        with torch.no_grad():
            for i, batch in tqdm(enumerate(data_loader), total=num_batches):
                if i >= num_batches:
                    break
                
                inputs, _ = batch
                inputs = inputs.to(self.device)
                
                self.ref_activations.clear()
                self.ref_model(inputs)
                
                # Process activations
                for name, act in self.ref_activations.items():
                    if name not in self.layer_metrics:
                        self.layer_metrics[name] = {'ref': [], 'int8': [], 'fp8': []}
                    
                    stats = self._compute_ref_stats(act)
                    self.layer_metrics[name]['ref'].append(stats)

    def collect_quant_metrics(self, data_loader, num_batches, quant_type):
        self.quant_model.eval()
        self.quant_model.to(self.device)
        
        # Update quantization type for all layers
        # We assume the model is already instantiated. We just change the q_type attribute.
        for module in self.quant_model.modules():
            if isinstance(module, QuantizedLayerMixin):
                module.q_type = quant_type
                # Also reset scales if needed? 
                # calibrate_weights should be called if we change q_type?
                # Yes, scales depend on q_type (bias, max_val mapping).
                # But we are in "simulate" mode usually.
                # GenericAdapter has _calibrate_model.
                # We should re-calibrate.
                pass
        
        # Re-calibrate weights
        # We need to call calibrate_weights on all modules
        for module in self.quant_model.modules():
            if hasattr(module, 'calibrate_weights'):
                module.calibrate_weights()
                
        print(f"Collecting {quant_type} metrics on {num_batches} batches...")
        
        key = 'int8' if 'int8' in quant_type else 'fp8'
        
        with torch.no_grad():
            for i, batch in tqdm(enumerate(data_loader), total=num_batches):
                if i >= num_batches:
                    break
                
                inputs, _ = batch
                inputs = inputs.to(self.device)
                
                # We need REF activations for comparison on the SAME input
                # So we run REF again? Or did we store them?
                # Storing all activations for all batches is memory intensive.
                # Better to run REF and QUANT in lockstep.
                
                # Run REF
                self.ref_activations.clear()
                self.ref_model(inputs)
                
                # Run QUANT
                self.quant_activations.clear()
                self.quant_model(inputs)
                
                # Compare
                for name, quant_act in self.quant_activations.items():
                    if name in self.ref_activations:
                        ref_act = self.ref_activations[name]
                        
                        if name not in self.layer_metrics:
                            # Should have been created in collect_ref_metrics if we ran it separately
                            # But if we run lockstep, we create it here.
                            self.layer_metrics[name] = {'ref': [], 'int8': [], 'fp8': []}
                        
                        stats = self._compute_compare_stats(ref_act, quant_act)
                        self.layer_metrics[name][key].append(stats)

    def run_lockstep_collection(self, data_loader, num_batches, modes=['ref', 'int8', 'fp8']):
        """
        Runs Ref, Int8, and FP8 in lockstep for each batch to save memory and ensure alignment.
        """
        self.ref_model.eval()
        self.quant_model.eval()
        self.ref_model.to(self.device)
        self.quant_model.to(self.device)
        
        print(f"Running lockstep collection for {modes} on {num_batches} batches...")
        
        with torch.no_grad():
            for i, batch in tqdm(enumerate(data_loader), total=num_batches):
                if i >= num_batches:
                    break
                
                inputs, _ = batch
                inputs = inputs.to(self.device)
                
                # 1. Run REF
                self.ref_activations.clear()
                self.ref_model(inputs)
                
                # Store REF stats
                for name, act in self.ref_activations.items():
                    if name not in self.layer_metrics:
                        self.layer_metrics[name] = {'ref': [], 'int8': [], 'fp8': []}
                    
                    # Only compute ref stats once (e.g. during first mode pass or just always)
                    # We append to list, so we should do it once per batch.
                    stats = self._compute_ref_stats(act)
                    self.layer_metrics[name]['ref'].append(stats)

                # 2. Run INT8
                if 'int8' in modes:
                    # Set to INT8
                    self._set_quant_type('int8')
                    self.quant_activations.clear()
                    self.quant_model(inputs)
                    
                    for name, quant_act in self.quant_activations.items():
                        if name in self.ref_activations:
                            stats = self._compute_compare_stats(self.ref_activations[name], quant_act)
                            self.layer_metrics[name]['int8'].append(stats)

                # 3. Run FP8
                if 'fp8' in modes:
                    # Set to FP8
                    self._set_quant_type('fp8_e4m3')
                    self.quant_activations.clear()
                    self.quant_model(inputs)
                    
                    for name, quant_act in self.quant_activations.items():
                        if name in self.ref_activations:
                            stats = self._compute_compare_stats(self.ref_activations[name], quant_act)
                            self.layer_metrics[name]['fp8'].append(stats)

    def _set_quant_type(self, q_type):
        for module in self.quant_model.modules():
            if isinstance(module, QuantizedLayerMixin):
                module.q_type = q_type
        
        # Recalibrate
        for module in self.quant_model.modules():
            if hasattr(module, 'calibrate_weights'):
                module.calibrate_weights()

    def aggregate_metrics(self) -> pd.DataFrame:
        # Average over batches
        rows = []
        for name, data in self.layer_metrics.items():
            row = {'layer': name}
            
            # Ref stats (avg)
            if data['ref']:
                ref_df = pd.DataFrame(data['ref'])
                for col in ref_df.columns:
                    row[col] = ref_df[col].mean()
            
            # Int8 stats
            if data['int8']:
                int8_df = pd.DataFrame(data['int8'])
                for col in int8_df.columns:
                    row[f"{col}_int8"] = int8_df[col].mean()
            
            # Fp8 stats
            if data['fp8']:
                fp8_df = pd.DataFrame(data['fp8'])
                for col in fp8_df.columns:
                    row[f"{col}_fp8"] = fp8_df[col].mean()
            
            rows.append(row)
        
        return pd.DataFrame(rows)

def visualize_metrics(df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Scatter: cos_int8 vs cos_fp8
    if 'cosine_int8' in df.columns and 'cosine_fp8' in df.columns:
        plt.figure(figsize=(10, 8))
        plt.scatter(df['cosine_int8'], df['cosine_fp8'], alpha=0.6)
        plt.plot([0.9, 1.0], [0.9, 1.0], 'r--', label='y=x')
        plt.xlabel('Cosine Similarity (INT8)')
        plt.ylabel('Cosine Similarity (FP8)')
        plt.title('Layer-wise Cosine Similarity: INT8 vs FP8')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'scatter_cosine.png'))
        plt.close()

    # 2. Bar plot: R = max_abs / rms
    if 'R' in df.columns:
        df_sorted = df.sort_values('R', ascending=False)
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(df_sorted)), df_sorted['R'])
        plt.xlabel('Layer Index (Sorted)')
        plt.ylabel('R = max_abs / rms')
        plt.title('Outlier Ratio (R) per Layer')
        plt.axhline(y=16, color='r', linestyle='--', label='Threshold (16)')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'bar_R.png'))
        plt.close()

def decide_precision(df: pd.DataFrame) -> Dict[str, str]:
    recommendations = {}
    
    for _, row in df.iterrows():
        layer = row['layer']
        
        # Decision Logic
        # Recommend FP8 if ANY:
        # • R > 16
        # • outlier_frac > 1e-4
        # • cos_int8 < 0.995 AND (cos_fp8 - cos_int8) > 0.002
        # • sqnr_int8_db < 25 AND (sqnr_fp8_db - sqnr_int8_db) > 3
        
        use_fp8 = False
        reasons = []
        
        if row.get('R', 0) > 16:
            use_fp8 = True
            reasons.append(f"R={row['R']:.1f}>16")
            
        if row.get('outlier_frac', 0) > 1e-4:
            use_fp8 = True
            reasons.append(f"Outliers={row['outlier_frac']:.1e}>1e-4")
            
        cos_int8 = row.get('cosine_int8', 1.0)
        cos_fp8 = row.get('cosine_fp8', 1.0)
        if cos_int8 < 0.995 and (cos_fp8 - cos_int8) > 0.002:
            use_fp8 = True
            reasons.append(f"CosGain={(cos_fp8-cos_int8):.4f}")
            
        sqnr_int8 = row.get('sqnr_db_int8', 100)
        sqnr_fp8 = row.get('sqnr_db_fp8', 100)
        if sqnr_int8 < 25 and (sqnr_fp8 - sqnr_int8) > 3:
            use_fp8 = True
            reasons.append(f"SQNRGain={(sqnr_fp8-sqnr_int8):.1f}dB")
            
        if use_fp8:
            recommendations[layer] = 'fp8_e4m3'
        else:
            recommendations[layer] = 'int8'
            
    return recommendations

def generate_mixed_config(base_config: dict, recommendations: dict, output_path: str):
    # Create layer_config dict
    layer_config = {}
    for layer, q_type in recommendations.items():
        layer_config[layer] = {
            'type': q_type,
            # We can also set input_format if needed, but usually we match weight format
            # 'input_format': q_type 
        }
        
    new_config = copy.deepcopy(base_config)
    
    # Ensure adapter config exists
    if 'adapter' not in new_config:
        new_config['adapter'] = {}
        
    new_config['adapter']['layer_config'] = layer_config
    # Set a valid default for global format (e.g. int8), as 'mixed' causes errors in calibration
    new_config['quantization']['format'] = 'int8' 
    new_config['meta']['quantization_mode'] = 'mixed' # Store intended mode in meta
    
    with open(output_path, 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False)
        
    print(f"Mixed-precision config saved to {output_path}")

def main():
    # Configuration
    # We can take args or hardcode for the experiment
    # Let's use a default base config and model
    
    base_config_path = os.path.join(PROJECT_ROOT, 'runspace/inputs/base_configs/advanced_full_config_int8.yaml')
    model_name = 'resnet18' # Default test model
    
    # Allow overriding via args
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        
    print(f"Starting Mixed-Precision Experiment for {model_name}...")
    
    # 1. Load Config & Data
    factory = ConfigFactory()
    # Create a dummy model dict to generate config
    model_info = [{'name': model_name, 'source': 'torchvision', 'weights': 'DEFAULT'}]
    configs = factory.create_configs(base_config_path, model_info)
    config = configs[0]
    
    runner = Runner()
    data_loader = runner.setup_data_loader(config)
    
    if data_loader is None:
        print("Error: Could not load data.")
        return

    # 2. Setup Models
    # Ref Model (FP32)
    print("Building Reference Model...")
    adapter = create_adapter(config)
    ref_model = adapter.build_reference_model()
    
    # Quant Model (Initial - will be mutated)
    print("Building Quant Model...")
    quant_model = adapter.model
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 3. Collect Metrics
    output_dir = os.path.join(PROJECT_ROOT, 'runspace/outputs', 'mixed_precision', model_name)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'layer_metrics.csv')
    
    if os.path.exists(csv_path):
        print(f"Loading existing metrics from {csv_path}...")
        df = pd.read_csv(csv_path)
    else:
        collector = MetricCollector(ref_model, quant_model, device)
        collector.register_hooks()
        
        # Run on ~512 samples (batch size 32 -> 16 batches)
        num_batches = 16
        collector.run_lockstep_collection(data_loader, num_batches, modes=['ref', 'int8', 'fp8'])
        
        collector.clear_hooks()
        
        # 4. Process & Visualize
        df = collector.aggregate_metrics()
        df.to_csv(csv_path, index=False)
        print(f"Metrics saved to {csv_path}")
    
    visualize_metrics(df, output_dir)
    
    # 5. Decide & Generate Config
    recommendations = decide_precision(df)
    
    # Print summary
    num_int8 = sum(1 for v in recommendations.values() if v == 'int8')
    num_fp8 = sum(1 for v in recommendations.values() if v == 'fp8_e4m3')
    print(f"Recommendations: INT8={num_int8}, FP8={num_fp8}")
    
    mixed_config_path = os.path.join(output_dir, 'mixed_config.yaml')
    generate_mixed_config(config, recommendations, mixed_config_path)
    
    # 6. Validation Run
    print("\nValidating Mixed-Precision Model...")
    
    # Load the new mixed config
    mixed_configs = factory.create_configs(mixed_config_path, model_info)
    mixed_config = mixed_configs[0]
    
    # Run evaluation
    res = runner.run_single(mixed_config, output_root=os.path.join(PROJECT_ROOT, 'runspace/outputs'))
    
    print("\n=== Validation Results ===")
    print(f"Top-1 Accuracy: {res['acc1']:.2f}%")
    print(f"Top-5 Accuracy: {res['acc5']:.2f}%")
    
    # Compare with baselines (from metrics or separate runs? metrics give layer-wise, not end-to-end)
    # Ideally we should run ALL-INT8 and ALL-FP8 end-to-end too for fair comparison.
    # But for now, we just report the mixed result.

if __name__ == "__main__":
    main()
