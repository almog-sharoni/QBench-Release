import os
import sys
import yaml
import glob
import json

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.core.config_factory import ConfigFactory
from runspace.core.runner import Runner

def load_models_list(path: str) -> list:
    with open(path, 'r') as f:
        if path.endswith('.json'):
            return json.load(f)
        elif path.endswith('.yaml') or path.endswith('.yml'):
            return yaml.safe_load(f)
        else:
            raise ValueError("Models list must be JSON or YAML")

def get_user_selection(options: list, name: str):
    print(f"\nAvailable {name}s:")
    for i, opt in enumerate(options):
        print(f"{i + 1}. {opt}")
    
    while True:
        try:
            choice = int(input(f"\nSelect a {name} (1-{len(options)}): "))
            if 1 <= choice <= len(options):
                return options[choice - 1]
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(options)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def main():
    print("=== QBench Interactive Run ===\n")
    
    # 1. List Base Configs
    base_configs_dir = os.path.join(PROJECT_ROOT, 'runspace/inputs/base_configs')
    if not os.path.exists(base_configs_dir):
        print(f"Error: Base configs directory not found at {base_configs_dir}")
        sys.exit(1)
        
    base_config_files = glob.glob(os.path.join(base_configs_dir, '*.yaml'))
    if not base_config_files:
        print(f"Error: No base config files found in {base_configs_dir}")
        sys.exit(1)
        
    base_config_names = [os.path.basename(f) for f in base_config_files]
    selected_base_config_name = get_user_selection(base_config_names, "Base Config")
    selected_base_config_path = os.path.join(base_configs_dir, selected_base_config_name)
    
    # 2. List Models
    models_path = os.path.join(PROJECT_ROOT, 'runspace/inputs/models.yaml')
    if not os.path.exists(models_path):
        print(f"Error: Models list not found at {models_path}")
        sys.exit(1)
        
    models = load_models_list(models_path)
    model_names = [m['name'] for m in models]
    selected_model_name = get_user_selection(model_names, "Model")
    
    # Find selected model config
    selected_model_config = next(m for m in models if m['name'] == selected_model_name)
    
    # 3. Generate Config
    print(f"\nGenerating configuration for {selected_model_name} using {selected_base_config_name}...")
    factory = ConfigFactory()
    # Create a single config list (factory returns a list)
    configs = factory.create_configs(selected_base_config_path, [selected_model_config], base_config_path=selected_base_config_path)
    
    if not configs:
        print("Error: Failed to generate configuration.")
        sys.exit(1)
        
    config = configs[0]
    
    # 4. Run
    print("\nStarting execution...")
    runner = Runner()
    result = runner.run_single(config)
    
    print("\n=== Execution Completed ===")
    print(f"Status: {result['status']}")
    if result['status'] == 'SUCCESS':
        print(f"Top-1 Accuracy: {result.get('acc1', 0):.2f}%")
        print(f"Top-5 Accuracy: {result.get('acc5', 0):.2f}%")
        print(f"Report: {result.get('report_path', 'N/A')}")
    else:
        print(f"Error: {result.get('exec_error', 'Unknown error')}")

if __name__ == "__main__":
    main()
