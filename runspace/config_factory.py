import yaml
import os
import copy
from typing import List, Dict, Any, Union

class ConfigFactory:
    """
    Factory class to generate configurations for different models based on a base configuration.
    """
    def __init__(self):
        pass

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Loads a YAML configuration file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def create_configs(self, base_config: Union[str, Dict[str, Any]], models: List[Dict[str, Any]], base_config_path: str = None) -> List[Dict[str, Any]]:
        """
        Generates a list of configurations by merging the base config with model-specific details.

        Args:
            base_config: Path to a base YAML config file or a dictionary containing the base config.
            models: A list of dictionaries, where each dictionary contains model-specific details
                    (e.g., name, source, weights, input_shape).
            base_config_path: Optional path to the base config file, for metadata injection.

        Returns:
            A list of complete configuration dictionaries.
        """
        if isinstance(base_config, str):
            base_config_data = self.load_config(base_config)
            if base_config_path is None:
                base_config_path = base_config
        else:
            base_config_data = base_config

        generated_configs = []

        for model in models:
            # Deep copy the base config to avoid modifying it for other models
            new_config = copy.deepcopy(base_config_data)

            # Ensure 'model' section exists
            if 'model' not in new_config:
                new_config['model'] = {}

            # Update model details
            # We expect 'model' dict in the input list to have keys like 'name', 'source', 'weights', etc.
            # These will override or add to the base config's model section.
            for key, value in model.items():
                new_config['model'][key] = value
            
            # Inject metadata
            if 'meta' not in new_config:
                new_config['meta'] = {}
            if base_config_path:
                new_config['meta']['base_config_path'] = os.path.abspath(base_config_path)

            generated_configs.append(new_config)

        return generated_configs

    def save_configs(self, configs: List[Dict[str, Any]], output_dir: str) -> List[str]:
        """
        Saves a list of configuration dictionaries to YAML files in the specified directory.
        
        Args:
            configs: List of configuration dictionaries.
            output_dir: Directory to save the generated config files.
            
        Returns:
            List of saved file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        saved_paths = []
        for config in configs:
            model_name = config.get('model', {}).get('name', 'unknown_model')
            base_name = os.path.basename(config.get('meta', {}).get('base_config_path', 'config')).replace('.yaml', '')
            
            # Create a unique filename combining model and base config name
            filename = f"{model_name}_{base_name}.yaml"
            filepath = os.path.join(output_dir, filename)
            
            # Inject the generated config path into metadata before saving
            if 'meta' not in config:
                config['meta'] = {}
            config['meta']['generated_config_path'] = os.path.abspath(filepath)
            
            with open(filepath, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            saved_paths.append(filepath)
            
        return saved_paths
