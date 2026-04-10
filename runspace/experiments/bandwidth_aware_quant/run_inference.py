import os
import sys
import csv

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.core.runner import Runner

def main():
    csv_path = os.path.join(os.path.dirname(__file__), "results/resnet18/l1/layer_errors.csv")
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return

    # Parse layer mapping
    layer_mapping = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer = row['layer']
            best_type = row['best_type']
            layer_mapping[layer] = {'format': best_type}

    print(f"Loaded mapping for {len(layer_mapping)} layers.")
    
    # Create configuration for the Runner
    config = {
        'model': {
            'name': 'resnet18',
            'weights': 'DEFAULT'
        },
        'adapter': {
            'type': 'generic',
            'input_quantization': False,
            'quantized_ops': ['-1'],  # Quantize all supported ops
        },
        'quantization': {
            'format': 'fp8_e1m6',     # Default format
            'layers': layer_mapping,  # Per-layer specific formats
            'weight_mode': 'chunk',  
            'weight_chunk_size': 128
        },
        'dataset': {
            'name': 'imagenet',
            'path': '/data/imagenet/val',  # Bound dynamically via Apptainer
            'batch_size': 128,
            'num_workers': 8
        },
        'evaluation': {
            'mode': 'evaluate',       # Just standard evaluation
            # 'max_batches': 10       # Let's run full inference
        },
        'output_name': 'bw_aware_custom_inference'
    }

    # Run inference
    runner = Runner()
    output_dir = os.path.join(PROJECT_ROOT, "runspace/experiments/bandwidth_aware_quant/results/inference_output")
    print(f"Running inference. Output will be saved to: {output_dir}")
    print(f"Config: {config}")
    
    try:
        results = runner.run_batch([config], output_root=output_dir)
        print("\n\nInference completed successfully!")
        
        # Manually extract output
        if results and isinstance(results, list):
            res = results[0]
            if isinstance(res, dict):
                print(f"Final Metrics:")
                print(f"  Accuracy (top-1): {res.get('acc1', 'N/A')}%")
                print(f"  Accuracy (top-5): {res.get('acc5', 'N/A')}%")
                print(f"  Certainty:        {res.get('certainty', 'N/A')}")
                
                # Also save to JSON
                import json
                out_path = os.path.join(output_dir, "metrics.json")
                with open(out_path, "w") as f:
                    json.dump(res, f, indent=4)
                print(f"Saved metrics dict to {out_path}")
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
