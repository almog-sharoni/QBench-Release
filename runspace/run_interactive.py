import os
import sys
import yaml
import glob
import json

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Set TORCH_HOME to project-local cache if not set, to avoid permission errors
# in environments (like Docker) where the home directory might not be writable.
if 'TORCH_HOME' not in os.environ:
    os.environ['TORCH_HOME'] = os.path.join(PROJECT_ROOT, '.cache', 'torch')

if 'PYTORCH_KERNEL_CACHE_PATH' not in os.environ:
    os.environ['PYTORCH_KERNEL_CACHE_PATH'] = os.path.join(PROJECT_ROOT, '.cache', 'torch', 'kernels')

from runspace.core.config_factory import ConfigFactory
from runspace.core.runner import Runner
from runspace.core.report_aggregator import ReportAggregator

def load_models_list(path: str) -> list:
    with open(path, 'r') as f:
        if path.endswith('.json'):
            return json.load(f)
        elif path.endswith('.yaml') or path.endswith('.yml'):
            return yaml.safe_load(f)
        else:
            raise ValueError("Models list must be JSON or YAML")

def get_user_selection(options: list, name: str) -> list:
    print(f"\nAvailable {name}s:")
    for i, opt in enumerate(options):
        print(f"{i + 1}. {opt}")
    
    while True:
        try:
            user_input = input(f"\nSelect {name}s (comma-separated, e.g. 1,3 or 'all'): ").strip()
            if user_input.lower() == 'all':
                return options
                
            selections = []
            parts = user_input.split(',')
            valid = True
            
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                    
                choice = int(part)
                if 1 <= choice <= len(options):
                    selections.append(options[choice - 1])
                else:
                    print(f"Invalid choice '{choice}'. Please enter numbers between 1 and {len(options)}.")
                    valid = False
                    break
            
            if valid and selections:
                return selections
            elif not selections:
                 print("No valid selection made.")
                 
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas.")

def main():
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run QBench Interactive")
    parser.add_argument("--graph-only", action="store_true", help="Only generate quantization graphs without running full evaluation/comparison.")
    args = parser.parse_args()

    print("=== QBench Interactive Run ===\n")
    if args.graph_only:
        print("Note: Running in GRAPH ONLY mode. Execution will be skipped.\n")
    
    # 1. List Base Configs
    base_configs_dir = os.path.join(PROJECT_ROOT, 'runspace/inputs/base_configs')
    if not os.path.exists(base_configs_dir):
        print(f"Error: Base configs directory not found at {base_configs_dir}")
        sys.exit(1)
        
    base_config_files = glob.glob(os.path.join(base_configs_dir, '*.yaml'))
    
    if base_config_files:
        base_config_names = sorted([os.path.basename(f) for f in base_config_files])
    else:
        print(f"Warning: No base config files found in {base_configs_dir}. Using default list.")
        base_config_names = [
            "advanced_full_config_fp8e4m3.yaml",
            "advanced_full_config_fp8e5m2.yaml",
            "advanced_full_config_int8.yaml",
            "slm_config_fp8e4m3.yaml",
            "tf32_accum_config.yaml"
        ]
    selected_base_config_names = get_user_selection(base_config_names, "Base Config")
    
    # 2. List Model Files
    inputs_dir = os.path.join(PROJECT_ROOT, 'runspace/inputs')
    # Find all yaml files in inputs dir, excluding base_configs directory
    model_files = []
    for f in glob.glob(os.path.join(inputs_dir, '*.yaml')):
        model_files.append(os.path.basename(f))
        
    if not model_files:
        print(f"Error: No model list files found in {inputs_dir}")
        sys.exit(1)
        
    selected_model_file = get_user_selection(model_files, "Models List File")
    # If user selected multiple (which get_user_selection supports), we just take the first one or merge?
    # get_user_selection returns a list.
    # For model list file, we probably just want one file to load models from, or we can load from all selected.
    # Let's assume user picks one for simplicity, or handle multiple.
    
    # Actually, get_user_selection returns a list. Let's support merging models from multiple files.
    
    all_models = []
    for m_file in selected_model_file:
        m_path = os.path.join(inputs_dir, m_file)
        all_models.extend(load_models_list(m_path))
        
    if not all_models:
        print("Error: No models found in selected files.")
        sys.exit(1)

    model_names = [m['name'] for m in all_models]
    # Remove duplicates if any
    model_names = sorted(list(set(model_names)))
    
    selected_model_names = get_user_selection(model_names, "Model")
    
    # Filter selected models
    # We need to handle potential duplicates in all_models list if multiple files have same model
    # We take the first occurrence
    selected_models = []
    seen_names = set()
    for m in all_models:
        if m['name'] in selected_model_names and m['name'] not in seen_names:
            selected_models.append(m)
            seen_names.add(m['name'])
    
    # 3. Generate Configs
    print(f"\nGenerating configurations...")
    factory = ConfigFactory()
    all_configs = []
    
    for base_config_name in selected_base_config_names:
        base_config_path = os.path.join(base_configs_dir, base_config_name)
        print(f"Processing {base_config_name}...")
        configs = factory.create_configs(base_config_path, selected_models, base_config_path=base_config_path)
        
        # Inject graph_only flag if set
        if args.graph_only:
            for cfg in configs:
                if 'evaluation' not in cfg:
                    cfg['evaluation'] = {}
                cfg['evaluation']['graph_only'] = True
                # Ensure graph generation is enabled
                cfg['evaluation']['generate_graph_svg'] = True
                
        all_configs.extend(configs)
    
    if not all_configs:
        print("Error: Failed to generate configuration.")
        sys.exit(1)
        
    # 3.5. Ask for number of batches (apply to all) - SKIP if graph_only
    if not args.graph_only:
        # We check the first config for default
        first_config = all_configs[0]
        current_batches = first_config.get('evaluation', {}).get('compare_batches', -1)
        if current_batches == -1:
            default_str = "All"
        else:
            default_str = str(current_batches)
            
        print(f"\nNumber of batches to run for all configs (default: {default_str}):")
        print("Press Enter to use default, enter a positive number, or -1 for all.")
        
        while True:
            batch_input = input("Batches: ").strip()
            if not batch_input:
                break
            try:
                new_batches = int(batch_input)
                if new_batches > 0 or new_batches == -1:
                    for config in all_configs:
                        if 'evaluation' not in config:
                            config['evaluation'] = {}
                        config['evaluation']['compare_batches'] = new_batches
                    
                    batch_msg = "ALL" if new_batches == -1 else str(new_batches)
                    print(f"Set number of batches to {batch_msg}")
                    break
                else:
                    print("Please enter a positive number or -1.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        
    # 3.6. Ask for Histogram Generation - SKIP if graph_only
    if not args.graph_only:
        print("\nGenerate Quantization Histograms?")
        print("Note: This will force execution mode to 'compare' and may be slower due to reference model execution.")
        
        hist_input = input("Generate Histograms? (y/N): ").strip().lower()
        if hist_input in ('y', 'yes'):
            print("Enabling histogram generation for all configurations...")
            for config in all_configs:
                if 'evaluation' not in config:
                    config['evaluation'] = {}
                config['evaluation']['save_histograms'] = True
                # Force compare mode as histograms require reference vs quantized comparison
                config['evaluation']['mode'] = 'compare'
        else:
            print("Using configuration defaults for histograms.")
        
    # 4. Run Batch
    print(f"\nStarting execution of {len(all_configs)} configurations...")
    runner = Runner()
    # We use run_batch which returns a list of results
    # We need to define an output root. We'll use the default 'runspace/outputs'
    output_root = os.path.join(PROJECT_ROOT, 'runspace/outputs')
    results = runner.run_batch(all_configs, output_root=output_root)
    
    print("\n=== Execution Completed ===")
    
    # 5. Aggregate Reports - Only if we have results and NOT graph_only (though Runner might return partial results)
    if len(results) > 0 and not args.graph_only:
        print("Aggregating results...")
        aggregator = ReportAggregator()
        summary_path = os.path.join(output_root, 'interactive_summary_report.csv')
        aggregator.aggregate(results, summary_path)
        
        # Generate Markdown version
        summary_md_path = os.path.splitext(summary_path)[0] + '.md'
        aggregator.aggregate(results, summary_md_path)
        
        print(f"Summary Report: {summary_path}")
        print(f"Markdown Report: {summary_md_path}")
    elif args.graph_only:
        print("Graph generation completed. Check output directories.")
    else:
        print("No results to aggregate.")

if __name__ == "__main__":
    main()
