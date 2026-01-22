import os
import sys
import torch
import yaml
import time
import gc
from typing import List, Dict, Any

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.adapters.adapter_factory import create_adapter
from src.eval.evaluator import Evaluator
from src.eval.metrics import MetricsEngine
from src.eval.comparator import LayerComparator
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

class Runner:
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Runner initialized on device: {self.device}")

    def setup_data_loader(self, config: dict):
        """Setup the data loader from config."""
        dataset_config = config.get('dataset', {})
        
        # Check for SLM adapter
        if config.get('adapter', {}).get('type') == 'slm':
            print("SLM Adapter detected. Loading wikitext-2 dataset...")
            try:
                from datasets import load_dataset
                # Load wikitext-2 raw test split
                dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
                # Filter out empty lines
                dataset = dataset.filter(lambda x: len(x['text'].strip()) > 0)
                print(f"Loaded {len(dataset)} samples from wikitext-2.")
            except Exception as e:
                print(f"Error loading wikitext-2: {e}")
                print("Falling back to dummy text dataset.")
                class DummyTextDataset(torch.utils.data.Dataset):
                    def __init__(self, length=100):
                        self.length = length
                        self.data = ["Hello, my name is", "The quick brown fox", "Once upon a time", "In a galaxy far far away"]
                    def __len__(self):
                        return self.length
                    def __getitem__(self, idx):
                        return {'text': self.data[idx % len(self.data)]}
                
                dataset = DummyTextDataset(length=100)

            batch_size = dataset_config.get('batch_size', 4) # Default to smaller batch for SLM
            return DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Get dataset path
        path = dataset_config.get('path', 'tests/data/imagenette2-320/val')
        if not os.path.isabs(path):
            data_dir = os.path.join(PROJECT_ROOT, path)
        else:
            data_dir = path
        
        # Standard ImageNet transforms
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        
        if not os.path.exists(data_dir):
             print(f"Warning: Data directory {data_dir} does not exist. Skipping data loading.")
             return None

        dataset = datasets.ImageFolder(data_dir, transform=transform)
        
        # Load ImageNet Class Index for mapping (copied from run_eval.py)
        import json
        index_path = os.path.join(PROJECT_ROOT, 'tests/data/imagenet_class_index.json')
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                class_index = json.load(f)
            
            wnid_to_idx = {v[0]: int(k) for k, v in class_index.items()}
            local_class_to_idx = dataset.class_to_idx
            idx_map = {}
            for wnid, local_idx in local_class_to_idx.items():
                if wnid in wnid_to_idx:
                    idx_map[local_idx] = wnid_to_idx[wnid]
            
            def target_transform(target):
                return idx_map.get(target, target)
            
            dataset.target_transform = target_transform
        
        batch_size = dataset_config.get('batch_size', 32)
        num_workers = dataset_config.get('num_workers', 0)
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    def run_single(self, config: Dict[str, Any], output_root: str = "runspace/outputs") -> Dict[str, Any]:
        """Runs a single configuration and returns the results."""
        model_config = config.get('model', {})
        model_name = model_config.get('name', 'unknown')
        quant_format = config.get('quantization', {}).get('format', 'fp8')
        
        # Check for mixed precision
        layer_configs = config.get('quantization', {}).get('layers', {})
        if layer_configs:
            unique_formats = set()
            for layer_name, layer_cfg in layer_configs.items():
                if isinstance(layer_cfg, dict) and 'format' in layer_cfg:
                    unique_formats.add(layer_cfg['format'])
            
            if len(unique_formats) > 1:
                quant_format = "mixed"
            elif len(unique_formats) == 1:
                # If all layers override to the same thing, report that thing
                quant_format = list(unique_formats)[0]
        
        meta = config.get('meta', {})
        
        print(f"--- Running {model_name} with {quant_format} ---")
        
        results = {
            'model_name': model_name,
            'quant_format': quant_format,
            'base_config_path': meta.get('base_config_path', 'N/A'),
            'generated_config_path': meta.get('generated_config_path', 'N/A'),
            'output_name': config.get('output_name', ''), # Custom identifier
            'status': 'FAILED',
            'acc1': 0.0,
            'acc5': 0.0,
            'certainty': 0.0,
            'ref_acc1': 0.0,
            'ref_acc5': 0.0,
            'ref_certainty': 0.0,
            'acc_drop': 0.0,
            'acc_drop': 0.0,
            'weight_comp_red': 0.0,
            'weight_comp_share': 0.0,
            'input_comp_red': 0.0,
            'input_comp_share': 0.0,
            'exec_error': None
        }

        try:
            # Setup data
            data_loader = self.setup_data_loader(config)
            if data_loader is None:
                results['error'] = "Data loader failed"
                return results

            # Create adapter
            adapter = create_adapter(config)
            model = adapter.model
            model.to(self.device)
            
            # Check for quantized layers
            quant_layer_count = 0
            for module in model.modules():
                # Check if module name contains 'Quant' (simple heuristic based on class name)
                if 'Quant' in module.__class__.__name__:
                    quant_layer_count += 1
            
            if quant_layer_count == 0:
                print(f"Warning: No quantized layers found for {model_name}")
                results['status'] = 'NO_QUANT'
            
            # Generate Output Directory
            base_config_path = meta.get('base_config_path', '')
            output_name = config.get('output_name', '')
            
            if output_name:
                # If explicit output_name is provided, use it directly under output_root or model dir
                # We want consistency: output_root / model_name / output_name 
                # (since find_optimal_layer_quant sets output_root = experiment/model)
                # But Runner standard is output_root/model/variant.
                # If output_root is already specific (like in the experiment script), we might want just output_root/output_name?
                # The experiment script passes output_root=model_dir.
                # So if we do output_root/output_name, it becomes .../resnet18/ref_fp32. This is what we want.
                output_dir = os.path.join(output_root, output_name)
            elif base_config_path:
                base_config_name = os.path.splitext(os.path.basename(base_config_path))[0]
                output_dir = os.path.join(output_root, base_config_name, model_name, quant_format.replace('_', ''))
            else:
                output_dir = os.path.join(output_root, model_name, quant_format.replace('_', ''))
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Save Config
            config_path = os.path.join(output_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            print(f"Config saved to {config_path}")

            # Generate Quantization Graph
            generate_graph = config.get('evaluation', {}).get('generate_graph_svg', True)
            if generate_graph:
                try:
                    from src.utils.graph_viz import generate_quantization_graph
                    graph_path = os.path.join(output_dir, "quant_graph.svg")
                    print(f"Generating quantization graph at {graph_path}...")
                    generate_quantization_graph(model, graph_path, model_name=model_name)
                except Exception as e:
                    print(f"Failed to generate graph: {e}")
            else:
                print("Skipping graph generation (disabled in config)")

            # Check evaluation mode
            eval_mode = config.get('evaluation', {}).get('mode', 'compare')

            if eval_mode == 'evaluate':
                print(f"Running in EVALUATE mode (Quantized Model Only)...")
                metrics_engine = MetricsEngine()
                evaluator = Evaluator(adapter, metrics_engine, device=self.device)
                
                # Run full evaluation
                eval_results = evaluator.evaluate(model, data_loader)
                
                results['acc1'] = eval_results.get('acc1', 0.0)
                results['acc5'] = eval_results.get('acc5', 0.0)
                results['certainty'] = eval_results.get('certainty', 0.0)
                
                if results['status'] != 'NO_QUANT':
                    results['status'] = 'SUCCESS'
                
            else:
                # COMPARE mode
                print(f"Running in COMPARE mode (Reference vs Quantized)...")
                
                # Build reference model
                ref_model = adapter.build_reference_model()
                ref_model.to(self.device)

                comparator = LayerComparator(
                    ref_model, 
                    model, 
                    model_name=model_name, 
                    quant_type=quant_format.replace('_', ''), 
                    adapter=adapter, 
                    device=self.device,
                    output_dir=output_dir,
                    save_histograms=config.get('evaluation', {}).get('save_histograms', False)
                )
                
                # Determine number of batches
                compare_batches = config.get('evaluation', {}).get('compare_batches', -1)
                if compare_batches == -1:
                    compare_batches = len(data_loader)
                    
                # Run Comparison (Single Pass)
                comparator.compare(data_loader, num_batches=compare_batches, global_metrics=None)
                
                # Retrieve metrics from comparator
                print("Retrieving metrics from comparator...")
                quant_metrics = comparator.quant_metrics.compute()
                ref_metrics = comparator.ref_metrics.compute()
                
                results['acc1'] = quant_metrics.get('acc1', 0.0)
                results['acc5'] = quant_metrics.get('acc5', 0.0)
                results['certainty'] = quant_metrics.get('certainty', 0.0)
                
                results['ref_acc1'] = ref_metrics.get('acc1', 0.0)
                results['ref_acc5'] = ref_metrics.get('acc5', 0.0)
                results['ref_certainty'] = ref_metrics.get('certainty', 0.0)
                
                results['acc_drop'] = results['ref_acc1'] - results['acc1']
                
                # Retrieve Compression Stats
                comp_stats = comparator.compression_tracker.get_stats()
                results['weight_comp_red'] = comp_stats['weight_compression_reduction']
                results['weight_comp_share'] = comp_stats['weight_share_of_total']
                results['input_comp_red'] = comp_stats['input_compression_reduction']
                results['input_comp_share'] = comp_stats['input_share_of_total']

                comparator.close()
                
                if results['status'] != 'NO_QUANT':
                    results['status'] = 'SUCCESS'
                    # Check for unquantized supported ops
                    if hasattr(comparator, 'unquantized_supported_count') and comparator.unquantized_supported_count > 0:
                        results['status'] = f"SUCCESS ({comparator.unquantized_supported_count} Unquantized)"
                        
                results['report_path'] = f"{output_dir}/comparison_report.txt"

        except Exception as e:
            print(f"Error running config for {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results['status'] = 'FAILED'
            results['exec_error'] = str(e)
        
        finally:
            print(f"Cleaning up memory for {model_name}...")
            if 'model' in locals(): del model
            if 'ref_model' in locals(): del ref_model
            if 'adapter' in locals(): del adapter
            if 'evaluator' in locals(): del evaluator
            if 'ref_evaluator' in locals(): del ref_evaluator
            if 'comparator' in locals(): del comparator
            if 'data_loader' in locals(): del data_loader
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        return results

    def run_batch(self, configs: List[Dict[str, Any]], output_root: str = "runspace/outputs") -> List[Dict[str, Any]]:
        """Runs a batch of configurations."""
        batch_results = []
        for i, config in enumerate(configs):
            print(f"Processing config {i+1}/{len(configs)}")
            res = self.run_single(config, output_root=output_root)
            batch_results.append(res)
        return batch_results
