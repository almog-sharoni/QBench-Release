import argparse
import os
import sys
import yaml
import json

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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

def main():
    parser = argparse.ArgumentParser(description='Run QBench Batch Evaluation')
    parser.add_argument('--base-configs', type=str, nargs='+', help='List of paths to base configuration YAMLs')
    parser.add_argument('--models-list', type=str, help='Path to models list (JSON/YAML)')
    parser.add_argument('--output-dir', type=str, default='runspace/outputs', help='Directory to save reports and configs')
    parser.add_argument('--summary-file', type=str, default='summary_report.csv', help='Filename for summary report')
    
    args = parser.parse_args()
    
    # Auto-discovery logic
    import glob
    
    # Resolve models list
    models_list_path = args.models_list
    if not models_list_path:
        default_models_path = os.path.join(PROJECT_ROOT, 'runspace/inputs/models.yaml')
        if os.path.exists(default_models_path):
            print(f"Auto-discovered models list: {default_models_path}")
            models_list_path = default_models_path
        else:
            print(f"Error: Models list not provided and not found at {default_models_path}")
            sys.exit(1)
            
    # Resolve base configs
    base_configs_paths = args.base_configs
    if not base_configs_paths:
        default_base_configs_dir = os.path.join(PROJECT_ROOT, 'runspace/inputs/base_configs')
        if os.path.exists(default_base_configs_dir):
            base_configs_paths = glob.glob(os.path.join(default_base_configs_dir, '*.yaml'))
            if base_configs_paths:
                print(f"Auto-discovered {len(base_configs_paths)} base configs in {default_base_configs_dir}")
            else:
                print(f"Error: No YAML files found in {default_base_configs_dir}")
                sys.exit(1)
        else:
             print(f"Error: Base configs not provided and directory {default_base_configs_dir} not found")
             sys.exit(1)

    factory = ConfigFactory()
    models = load_models_list(models_list_path)
    
    all_configs = []
    
    # 1. Generate Configs for each base config
    print("Generating configurations...")
    config_output_dir = os.path.join(args.output_dir, 'configs')
    
    for base_config_path in base_configs_paths:
        print(f"Processing base config: {base_config_path}")
        configs = factory.create_configs(base_config_path, models, base_config_path=base_config_path)
        
        # Save generated configs
        factory.save_configs(configs, config_output_dir)
        all_configs.extend(configs)
        
    print(f"Generated {len(all_configs)} total configurations in {config_output_dir}")
    
    # 2. Run Batch
    print("Starting batch execution...")
    runner = Runner()
    results = runner.run_batch(all_configs, output_root=args.output_dir)
    
    # 3. Aggregate Reports
    print("Aggregating results...")
    aggregator = ReportAggregator()
    summary_path = os.path.join(args.output_dir, args.summary_file)
    aggregator.aggregate(results, summary_path)
    
    print("Batch run completed.")

if __name__ == "__main__":
    main()
