import os
import sys
import yaml
import glob
import json
import shlex
from datetime import datetime, timezone

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

SCRIPT_REL_PATH = os.path.relpath(__file__, PROJECT_ROOT)
DEFAULT_COMMAND_LOG_PATH = os.path.join(PROJECT_ROOT, 'runspace', 'interactive_commands.log')
PYTHON_COMMAND = os.path.basename(sys.executable) or 'python'

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

def _parse_csv_arg(value: str) -> list:
    """Split a comma-separated CLI string into a stripped list."""
    return [v.strip() for v in value.split(',') if v.strip()]


def _csv_arg(values: list) -> str:
    return ",".join(values)


def _command_to_argv(command: str) -> list:
    """Extract run_interactive.py arguments from a saved shell command."""
    parts = shlex.split(command)
    for idx, part in enumerate(parts):
        if os.path.basename(part) == 'run_interactive.py':
            return parts[idx + 1:]
    return parts


def _load_command_history(path: str) -> list:
    if not path or not os.path.exists(path):
        return []

    commands = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                command = entry.get('command')
            except json.JSONDecodeError:
                command = line
            if command:
                commands.append(command)
    return commands


def _append_command_history(path: str, command: str):
    log_dir = os.path.dirname(path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    entry = {
        'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'command': command,
    }
    with open(path, 'a') as f:
        f.write(json.dumps(entry, sort_keys=True) + '\n')


def _print_command_history(commands: list, limit: int = None):
    if not commands:
        print("No saved interactive commands found.")
        return

    shown = commands if limit is None else commands[-limit:]
    start = len(commands) - len(shown) + 1
    print("\nSaved interactive commands:")
    for offset, command in enumerate(shown):
        print(f"{start + offset}. {command}")


def _select_command_from_history(commands: list) -> str:
    if not commands:
        print("No saved interactive commands found.")
        return None

    _print_command_history(commands, limit=10)
    print("\nEnter a command number to run it again, 'last' for the newest command,")
    print("or press Enter to start a new interactive run.")
    while True:
        choice = input("Command: ").strip().lower()
        if not choice:
            return None
        if choice in ('last', 'l'):
            return commands[-1]
        try:
            index = int(choice)
        except ValueError:
            print("Invalid input. Enter a command number, 'last', or press Enter.")
            continue
        if 1 <= index <= len(commands):
            return commands[index - 1]
        print(f"Invalid choice '{index}'. Please enter a number between 1 and {len(commands)}.")


def _has_run_selection_args(args) -> bool:
    return any([
        args.base_config,
        args.models_file,
        args.models,
        args.batches is not None,
        args.default_batches,
        args.workers is not None,
        args.histograms,
        args.no_histograms,
        args.graph_only,
    ])


def _build_replay_command(
    selected_base_config_names: list,
    selected_model_files: list,
    selected_model_names: list,
    batch_arg,
    histogram_arg,
    workers,
    graph_only: bool,
) -> str:
    command = [
        PYTHON_COMMAND,
        SCRIPT_REL_PATH,
        '--base-config',
        _csv_arg(selected_base_config_names),
        '--models-file',
        _csv_arg(selected_model_files),
        '--models',
        _csv_arg(selected_model_names),
    ]
    if graph_only:
        command.append('--graph-only')
    elif batch_arg == 'default':
        command.append('--default-batches')
    elif batch_arg is not None:
        command.extend(['--batches', str(batch_arg)])

    if workers is not None:
        command.extend(['--workers', str(workers)])

    if histogram_arg == 'enabled':
        command.append('--histograms')
    elif histogram_arg == 'defaults':
        command.append('--no-histograms')

    return shlex.join(command)


def _resolve_selection(cli_values, options: list, name: str) -> list:
    """
    Use cli_values if provided, otherwise fall back to interactive prompt.
    cli_values: None (not supplied), 'all', or a list of names.
    """
    if cli_values is None:
        return get_user_selection(options, name)

    if cli_values == 'all':
        return options

    invalid = [v for v in cli_values if v not in options]
    if invalid:
        print(f"Error: Unknown {name}(s): {invalid}")
        print(f"Available: {options}")
        sys.exit(1)

    return cli_values


def _build_parser():
    import argparse
    parser = argparse.ArgumentParser(
        description="Run QBench. All arguments are optional; omitting any triggers an interactive prompt for that step.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fully interactive (original behaviour)
  python run_interactive.py

  # Non-interactive: one base config, one model file, all models
  python run_interactive.py --base-config fp8e4m3.yaml --models-file imagenet_models.yaml --models all

  # Non-interactive: specific models, 50 batches
  python run_interactive.py --base-config fp8e4m3.yaml --models-file imagenet_models.yaml --models resnet18,resnet50 --batches 50

  # Multiple base configs
  python run_interactive.py --base-config fp8e4m3.yaml,fp8e5m2.yaml --models-file imagenet_models.yaml --models all

  # Run the last saved resolved interactive command
  python run_interactive.py --last-command
""",
    )
    parser.add_argument(
        "--base-config",
        metavar="NAME[,NAME...]",
        help="Base config filename(s) from inputs/base_configs/, comma-separated or 'all'.",
    )
    parser.add_argument(
        "--models-file",
        metavar="FILE[,FILE...]",
        help="Model list YAML filename(s) from inputs/, comma-separated or 'all'.",
    )
    parser.add_argument(
        "--models",
        metavar="NAME[,NAME...]",
        help="Model name(s) to run, comma-separated or 'all'.",
    )
    parser.add_argument(
        "--batches",
        type=int,
        metavar="N",
        help="Number of batches (-1 for all). Skips the batches prompt.",
    )
    parser.add_argument(
        "--default-batches",
        action="store_true",
        help="Use each config's default batch count. Skips the batches prompt.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        metavar="N",
        help="Override dataset.num_workers for all generated configs.",
    )
    parser.add_argument(
        "--histograms",
        action="store_true",
        default=None,
        help="Enable histogram generation (skips the histogram prompt).",
    )
    parser.add_argument(
        "--no-histograms",
        action="store_true",
        help="Use configuration defaults for histogram generation (skips the histogram prompt).",
    )
    parser.add_argument(
        "--graph-only",
        action="store_true",
        help="Only generate quantization graphs, skip evaluation.",
    )
    parser.add_argument(
        "--command-log",
        default=DEFAULT_COMMAND_LOG_PATH,
        help="Path to the resolved interactive command history log.",
    )
    parser.add_argument(
        "--list-commands",
        action="store_true",
        help="List saved resolved interactive commands and exit.",
    )
    parser.add_argument(
        "--choose-command",
        action="store_true",
        help="Choose a saved resolved interactive command to run.",
    )
    parser.add_argument(
        "--last-command",
        "--rerun-last",
        dest="last_command",
        action="store_true",
        help="Run the newest saved resolved interactive command.",
    )
    parser.add_argument(
        "--no-command-prompt",
        action="store_true",
        help="Do not offer saved commands before a fully interactive run.",
    )
    return parser


def main(argv=None, allow_command_prompt=True):
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Normalise histogram flag
    if args.histograms and args.no_histograms:
        parser.error("--histograms and --no-histograms are mutually exclusive.")
    if args.batches is not None and args.default_batches:
        parser.error("--batches and --default-batches are mutually exclusive.")
    histogram_cli = True if args.histograms else (False if args.no_histograms else None)

    command_history = _load_command_history(args.command_log)
    if args.list_commands:
        _print_command_history(command_history)
        return

    if args.last_command:
        if not command_history:
            print(f"No saved interactive commands found in {args.command_log}.")
            sys.exit(1)
        command = command_history[-1]
        replay_argv = _command_to_argv(command)
        if args.command_log != DEFAULT_COMMAND_LOG_PATH and '--command-log' not in replay_argv:
            replay_argv.extend(['--command-log', args.command_log])
        print(f"Running last saved command:\n{command}\n")
        return main(replay_argv, allow_command_prompt=False)

    if args.choose_command:
        command = _select_command_from_history(command_history)
        if command is not None:
            replay_argv = _command_to_argv(command)
            if args.command_log != DEFAULT_COMMAND_LOG_PATH and '--command-log' not in replay_argv:
                replay_argv.extend(['--command-log', args.command_log])
            print(f"Running saved command:\n{command}\n")
            return main(replay_argv, allow_command_prompt=False)

    if (
        allow_command_prompt
        and not args.no_command_prompt
        and not _has_run_selection_args(args)
        and command_history
    ):
        command = _select_command_from_history(command_history)
        if command is not None:
            replay_argv = _command_to_argv(command)
            if args.command_log != DEFAULT_COMMAND_LOG_PATH and '--command-log' not in replay_argv:
                replay_argv.extend(['--command-log', args.command_log])
            print(f"Running saved command:\n{command}\n")
            return main(replay_argv, allow_command_prompt=False)

    print("=== QBench Interactive Run ===\n")
    if args.graph_only:
        print("Note: Running in GRAPH ONLY mode. Execution will be skipped.\n")

    # 1. Base Configs
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
            "tf32_accum_config.yaml",
        ]

    cli_base = None if args.base_config is None else (
        'all' if args.base_config.strip().lower() == 'all' else _parse_csv_arg(args.base_config)
    )
    selected_base_config_names = _resolve_selection(cli_base, base_config_names, "Base Config")

    # 2. Model Files
    inputs_dir = os.path.join(PROJECT_ROOT, 'runspace/inputs')
    model_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(inputs_dir, '*.yaml'))])
    if not model_files:
        print(f"Error: No model list files found in {inputs_dir}")
        sys.exit(1)

    cli_mf = None if args.models_file is None else (
        'all' if args.models_file.strip().lower() == 'all' else _parse_csv_arg(args.models_file)
    )
    selected_model_files = _resolve_selection(cli_mf, model_files, "Models List File")

    all_models = []
    for m_file in selected_model_files:
        m_path = os.path.join(inputs_dir, m_file)
        all_models.extend(load_models_list(m_path))

    if not all_models:
        print("Error: No models found in selected files.")
        sys.exit(1)

    model_names = sorted(list({m['name'] for m in all_models}))

    cli_models = None if args.models is None else (
        'all' if args.models.strip().lower() == 'all' else _parse_csv_arg(args.models)
    )
    selected_model_names = _resolve_selection(cli_models, model_names, "Model")

    selected_models = []
    seen_names = set()
    for m in all_models:
        if m['name'] in selected_model_names and m['name'] not in seen_names:
            selected_models.append(m)
            seen_names.add(m['name'])

    from runspace.core.config_factory import ConfigFactory
    from runspace.core.runner import Runner
    from runspace.core.report_aggregator import ReportAggregator

    # 3. Generate Configs
    print(f"\nGenerating configurations...")
    factory = ConfigFactory()
    all_configs = []

    for base_config_name in selected_base_config_names:
        base_config_path = os.path.join(base_configs_dir, base_config_name)
        print(f"Processing {base_config_name}...")
        configs = factory.create_configs(base_config_path, selected_models, base_config_path=base_config_path)

        if args.graph_only:
            for cfg in configs:
                cfg.setdefault('evaluation', {})
                cfg['evaluation']['graph_only'] = True
                cfg['evaluation']['generate_graph_svg'] = True

        all_configs.extend(configs)

    if not all_configs:
        print("Error: Failed to generate configuration.")
        sys.exit(1)

    # 3.4. Workers override
    if args.workers is not None:
        for config in all_configs:
            config.setdefault('dataset', {})
            config['dataset']['num_workers'] = args.workers
        print(f"dataset.num_workers set to {args.workers} (from --workers).")

    # 3.5. Batches
    batch_arg = None
    if not args.graph_only:
        if args.default_batches:
            batch_arg = 'default'
            print("Using each configuration's default batch count (from --default-batches).")
        elif args.batches is not None:
            new_batches = args.batches
            for config in all_configs:
                config.setdefault('evaluation', {})
                config['evaluation']['compare_batches'] = new_batches
                config['evaluation']['max_batches'] = new_batches
            batch_msg = "ALL" if new_batches == -1 else str(new_batches)
            print(f"Batches set to {batch_msg} (from --batches).")
            batch_arg = new_batches
        else:
            first_config = all_configs[0]
            current_batches = first_config.get('evaluation', {}).get('compare_batches', -1)
            default_str = "All" if current_batches == -1 else str(current_batches)
            print(f"\nNumber of batches to run for all configs (default: {default_str}):")
            print("Press Enter to use default, enter a positive number, or -1 for all.")
            while True:
                batch_input = input("Batches: ").strip()
                if not batch_input:
                    batch_arg = 'default'
                    break
                try:
                    new_batches = int(batch_input)
                    if new_batches > 0 or new_batches == -1:
                        for config in all_configs:
                            config.setdefault('evaluation', {})
                            config['evaluation']['compare_batches'] = new_batches
                            config['evaluation']['max_batches'] = new_batches
                        batch_msg = "ALL" if new_batches == -1 else str(new_batches)
                        print(f"Set number of batches to {batch_msg}")
                        batch_arg = new_batches
                        break
                    else:
                        print("Please enter a positive number or -1.")
                except ValueError:
                    print("Invalid input. Please enter a number.")

    # 3.6. Histograms
    histogram_arg = None
    if not args.graph_only:
        if histogram_cli is not None:
            enable_histograms = histogram_cli
            print(f"Histogram generation {'enabled' if enable_histograms else 'disabled'} (from CLI flag).")
        else:
            print("\nGenerate Quantization Histograms?")
            print("Note: This will force execution mode to 'compare' and may be slower.")
            hist_input = input("Generate Histograms? (y/N): ").strip().lower()
            enable_histograms = hist_input in ('y', 'yes')

        if enable_histograms:
            print("Enabling histogram generation for all configurations...")
            for config in all_configs:
                config.setdefault('evaluation', {})
                config['evaluation']['save_histograms'] = True
                config['evaluation']['mode'] = 'compare'
            histogram_arg = 'enabled'
        else:
            print("Using configuration defaults for histograms.")
            histogram_arg = 'defaults'

    replay_command = _build_replay_command(
        selected_base_config_names,
        selected_model_files,
        selected_model_names,
        batch_arg,
        histogram_arg,
        args.workers,
        args.graph_only,
    )
    _append_command_history(args.command_log, replay_command)
    print(f"\nSaved replay command to {args.command_log}:")
    print(replay_command)
        
    # 4. Run Configs Sequentially (explicit loop; runner handles single-run only)
    print(f"\nStarting execution of {len(all_configs)} configurations...")
    runner = Runner()
    output_root = os.path.join(PROJECT_ROOT, 'runspace/outputs')
    results = []
    total = len(all_configs)
    for idx, cfg in enumerate(all_configs, start=1):
        out_name = cfg.get('output_name', f'run_{idx}')
        print(f"Running config {idx}/{total}: {out_name}")
        res = runner.run_single_logged(cfg, output_root=output_root)
        results.append(res)
    
    print("\n=== Execution Completed ===")
    
    # 5. Aggregate Reports - Only if we have results and NOT graph_only (though Runner might return partial results)
    if len(results) > 0 and not args.graph_only:
        print("Aggregating results...")
        aggregator = ReportAggregator()
        base_stem = os.path.join(output_root, 'interactive_summary_report')
        aggregator.aggregate(results, base_stem + '.csv')
        aggregator.aggregate(results, base_stem + '.md')

        stem = base_stem
        has_cls = any(r.get('adapter_type') != 'feature_matching' for r in results)
        has_fm  = any(r.get('adapter_type') == 'feature_matching' for r in results)
        if has_cls:
            print(f"Classification CSV:  {stem}_classification.csv")
            print(f"Classification MD:   {stem}_classification.md")
        if has_fm:
            print(f"Feature Matching CSV: {stem}_fm.csv")
            print(f"Feature Matching MD:  {stem}_fm.md")
    elif args.graph_only:
        print("Graph generation completed. Check output directories.")
    else:
        print("No results to aggregate.")

if __name__ == "__main__":
    main()
