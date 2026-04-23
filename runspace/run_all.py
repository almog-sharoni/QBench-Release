import argparse
import os
import sys
import yaml
import json
import logging
from datetime import datetime

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if 'TORCH_HOME' not in os.environ:
    os.environ['TORCH_HOME'] = os.path.join(PROJECT_ROOT, '.cache', 'torch')

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

def setup_logging(output_dir: str) -> str:
    """Set up logging to both stdout and a timestamped log file."""
    logs_dir = os.path.join(output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(logs_dir, f'run_all_{timestamp}.log')

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter('%(message)s')

    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    # Redirect print() to the logger so all existing prints land in the file too.
    class _PrintToLogger:
        def __init__(self, level):
            self._level = level
            self._buf = ''
        def write(self, msg):
            self._buf += msg
            while '\n' in self._buf:
                line, self._buf = self._buf.split('\n', 1)
                logging.log(self._level, line)
        def flush(self):
            if self._buf:
                logging.log(self._level, self._buf)
                self._buf = ''
        def fileno(self):
            raise io.UnsupportedOperation('fileno')

    import io
    sys.stdout = _PrintToLogger(logging.INFO)
    sys.stderr = _PrintToLogger(logging.WARNING)

    return log_path

def main():
    parser = argparse.ArgumentParser(description='Run QBench Batch Evaluation')
    parser.add_argument('--base-configs', type=str, nargs='+', help='List of paths to base configuration YAMLs')
    parser.add_argument('--models-list', type=str, help='Path to models list (JSON/YAML)')
    parser.add_argument('--output-dir', type=str, default='runspace/outputs', help='Directory to save reports and configs')
    parser.add_argument('--summary-file', type=str, default='summary_report.csv', help='Filename for summary report')
    parser.add_argument('--batches', type=int, default=None, help='Max evaluation batches per run (-1 for all)')
    parser.add_argument('--stop-on-error', action='store_true', help='Abort after the first failed run')
    parser.add_argument('--workers', type=int, default=None, help='Override dataset.num_workers for all configs')

    args = parser.parse_args()

    log_path = setup_logging(args.output_dir)
    # Print after logging is wired so it goes to the file too.
    print(f"Logging to {log_path}")

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
        print(f"Processing {os.path.basename(base_config_path)}...")
        configs = factory.create_configs(base_config_path, models, base_config_path=base_config_path)

        # Apply --batches override
        if args.batches is not None:
            for cfg in configs:
                cfg.setdefault('evaluation', {})
                cfg['evaluation']['compare_batches'] = args.batches
                cfg['evaluation']['max_batches'] = args.batches

        # Apply --workers override
        if args.workers is not None:
            for cfg in configs:
                cfg.setdefault('dataset', {})
                cfg['dataset']['num_workers'] = args.workers

        # Save generated configs
        factory.save_configs(configs, config_output_dir)
        all_configs.extend(configs)

    print(f"Generated {len(all_configs)} total configurations in {config_output_dir}")

    # 2. Run Configs Sequentially (explicit loop; runner handles single-run only)
    print(f"\nStarting execution of {len(all_configs)} configurations...")
    runner = Runner()
    results = []
    total = len(all_configs)
    for idx, cfg in enumerate(all_configs, start=1):
        out_name = cfg.get('output_name', f'run_{idx}')
        print(f"Running config {idx}/{total}: {out_name}")
        res = runner.run_single_logged(cfg, output_root=args.output_dir)
        results.append(res)
        if args.stop_on_error and res.get('status') == 'FAILED':
            print(f"Stopping after first error (--stop-on-error): {res.get('exec_error', 'unknown error')}")
            break

    # 3. Aggregate Reports
    print("Aggregating results...")
    aggregator = ReportAggregator()
    summary_path = os.path.join(args.output_dir, args.summary_file)
    aggregator.aggregate(results, summary_path)

    summary_md_path = os.path.splitext(summary_path)[0] + '.md'
    aggregator.aggregate(results, summary_md_path)

    print(f"Run completed. Log saved to {log_path}")

if __name__ == "__main__":
    main()
