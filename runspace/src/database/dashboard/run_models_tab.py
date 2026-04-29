with tab_runner:
    st.markdown("""
    <div class="dashboard-hero">
        <div class="dashboard-hero__eyebrow">Evaluation · Launcher</div>
        <h1>Run Models</h1>
        <p>Choose the same options as the interactive runner, then start a non-interactive QBench run from the dashboard.</p>
    </div>
    """, unsafe_allow_html=True)

    import glob
    import html
    import re
    import shlex
    import subprocess
    import time

    import yaml

    RUNNER_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "runspace/outputs/dashboard_runs")
    RUNNER_SCRIPT = os.path.join(PROJECT_ROOT, "runspace/run_interactive.py")
    BASE_CONFIGS_DIR = os.path.join(PROJECT_ROOT, "runspace/inputs/base_configs")
    INPUTS_DIR = os.path.join(PROJECT_ROOT, "runspace/inputs")

    def _dashboard_runner_list_files(folder, pattern="*.yaml"):
        return sorted(os.path.basename(path) for path in glob.glob(os.path.join(folder, pattern)))

    def _dashboard_runner_load_models(model_files):
        models = []
        seen = set()
        for model_file in model_files:
            model_path = os.path.join(INPUTS_DIR, model_file)
            try:
                with open(model_path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f) or []
            except Exception as exc:
                st.error(f"Could not load `{model_file}`: {exc}")
                continue

            if not isinstance(loaded, list):
                st.warning(f"`{model_file}` does not contain a list of models.")
                continue

            for item in loaded:
                if not isinstance(item, dict) or not item.get("name"):
                    continue
                name = str(item["name"])
                if name not in seen:
                    models.append(name)
                    seen.add(name)
        return sorted(models)

    def _dashboard_runner_safe_name(name):
        safe = []
        for char in str(name or "").strip():
            if char.isalnum() or char in ("-", "_"):
                safe.append(char)
            elif char.isspace() or char == ".":
                safe.append("_")
        safe_name = "".join(safe).strip("_")
        return safe_name or "dashboard_config"

    def _dashboard_runner_load_yaml(path):
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        return loaded or {}

    def _dashboard_runner_dump_yaml(data):
        return yaml.safe_dump(data, sort_keys=False, default_flow_style=False)

    def _dashboard_runner_split_csv(value):
        return [item.strip() for item in str(value or "").split(",") if item.strip()]

    def _dashboard_runner_csv_from_list(value, default=None):
        if isinstance(value, list):
            return ",".join(str(item) for item in value)
        if value is None:
            return ",".join(default or [])
        return str(value)

    def _dashboard_runner_build_config_from_form(template_config, form_values):
        config = template_config.copy() if isinstance(template_config, dict) else {}
        config = yaml.safe_load(_dashboard_runner_dump_yaml(config)) or {}

        config.setdefault("adapter", {})
        config.setdefault("quantization", {})
        config.setdefault("dataset", {})
        config.setdefault("evaluation", {})
        config.setdefault("meta", {})

        config["adapter"]["type"] = "generic"
        config["adapter"]["quantize_first_layer"] = bool(form_values["quantize_first_layer"])
        config["adapter"]["weight_quantization"] = bool(form_values["weight_quantization"])
        config["adapter"]["input_quantization"] = bool(form_values["input_quantization"])
        config["adapter"]["quantized_ops"] = _dashboard_runner_split_csv(form_values["quantized_ops"]) or ["all"]
        excluded_ops = _dashboard_runner_split_csv(form_values["excluded_ops"])
        config["adapter"]["excluded_ops"] = excluded_ops if excluded_ops else [""]

        config["quantization"]["format"] = form_values["weight_format"]
        config["quantization"]["input_format"] = form_values["input_format"]
        config["quantization"]["mode"] = form_values["input_mode"]
        config["quantization"]["chunk_size"] = int(form_values["input_chunk_size"])
        config["quantization"]["weight_mode"] = form_values["weight_mode"]
        config["quantization"]["weight_chunk_size"] = int(form_values["weight_chunk_size"])
        config["quantization"]["calib_method"] = form_values["calib_method"]
        config["quantization"]["rounding"] = form_values["rounding"]

        config["dataset"]["name"] = form_values["dataset_name"]
        config["dataset"]["path"] = form_values["dataset_path"]
        config["dataset"]["batch_size"] = int(form_values["batch_size"])
        config["dataset"]["num_workers"] = int(form_values["num_workers"])

        config["evaluation"]["mode"] = form_values["evaluation_mode"]
        config["evaluation"]["compare_batches"] = int(form_values["compare_batches"])
        config["evaluation"]["max_batches"] = int(form_values["compare_batches"])
        config["evaluation"]["generate_graph_svg"] = bool(form_values["generate_graph_svg"])
        config["evaluation"]["save_histograms"] = bool(form_values["save_histograms"])

        config["meta"]["created_by"] = "dashboard"
        config["meta"]["dashboard_config_name"] = form_values["config_name"]
        config["meta"]["template_config"] = form_values["template_config"]
        return config

    def _dashboard_runner_save_generated_config(config_name, config_data):
        safe_name = _dashboard_runner_safe_name(config_name)
        filename = f"{safe_name}.yaml"
        path = os.path.join(BASE_CONFIGS_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(_dashboard_runner_dump_yaml(config_data))
        return filename, path

    def _dashboard_runner_current_process():
        process = st.session_state.get("dashboard_runner_process")
        if process is not None and process.poll() is not None:
            st.session_state["dashboard_runner_returncode"] = process.returncode
            st.session_state["dashboard_runner_process"] = None
            return None
        return process

    def _dashboard_runner_read_log(path):
        if not path or not os.path.exists(path):
            return ""
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    def _dashboard_runner_lines(log_text):
        return log_text.splitlines() if log_text else []

    def _dashboard_runner_parse_progress(log_text):
        progress = {
            "total_configs": None,
            "current_config": 0,
            "current_name": "",
            "completed": False,
            "tqdm_percent": None,
            "tqdm_current": None,
            "tqdm_total": None,
            "tqdm_line": "",
        }
        if not log_text:
            return progress

        normalized_log = log_text.replace("\r", "\n")
        for line in normalized_log.splitlines():
            start_match = re.search(r"Starting execution of (\d+) configurations", line)
            if start_match:
                progress["total_configs"] = int(start_match.group(1))

            run_match = re.search(r"Running config (\d+)/(\d+):\s*(.*)", line)
            if run_match:
                progress["current_config"] = int(run_match.group(1))
                progress["total_configs"] = int(run_match.group(2))
                progress["current_name"] = run_match.group(3).strip()

            tqdm_match = re.search(r"(\d{1,3})%\|.*?\|\s*([0-9]+)/([0-9]+)", line)
            if tqdm_match:
                progress["tqdm_percent"] = min(100, int(tqdm_match.group(1)))
                progress["tqdm_current"] = int(tqdm_match.group(2))
                progress["tqdm_total"] = int(tqdm_match.group(3))
                progress["tqdm_line"] = line.strip()

            if "=== Execution Completed ===" in line:
                progress["completed"] = True

        if progress["completed"] and progress["total_configs"]:
            progress["current_config"] = progress["total_configs"]
        return progress

    def _dashboard_runner_render_loading(progress, is_active):
        total_configs = progress["total_configs"] or 0
        current_config = progress["current_config"] or 0
        current_name = progress["current_name"] or "Preparing run"
        completed = bool(progress["completed"])
        tqdm_percent = progress["tqdm_percent"]

        if completed:
            overall_percent = 100
            status_text = "Run completed"
            bar_class = "dashboard-runner-bar dashboard-runner-bar--done"
        elif total_configs > 0:
            completed_configs = max(0, current_config - 1)
            step_fraction = (tqdm_percent / 100.0) if tqdm_percent is not None else (0.12 if is_active else 0.0)
            overall_percent = int(min(99, max(0, ((completed_configs + step_fraction) / total_configs) * 100)))
            status_text = f"Running config {current_config}/{total_configs}"
            bar_class = "dashboard-runner-bar"
        elif is_active:
            overall_percent = 33
            status_text = "Starting runner"
            bar_class = "dashboard-runner-bar dashboard-runner-bar--indeterminate"
        else:
            overall_percent = 0
            status_text = "Idle"
            bar_class = "dashboard-runner-bar"

        safe_status = html.escape(status_text)
        safe_name = html.escape(current_name)
        safe_tqdm = html.escape(progress["tqdm_line"] or "Waiting for batch progress...")
        shimmer_width = max(8, min(100, overall_percent))

        st.markdown(f"""
        <style>
        @keyframes dashboardRunnerShimmer {{
            0% {{ transform: translateX(-120%); }}
            100% {{ transform: translateX(120%); }}
        }}
        @keyframes dashboardRunnerPulse {{
            0%, 100% {{ opacity: 0.58; transform: scale(1); }}
            50% {{ opacity: 1; transform: scale(1.08); }}
        }}
        .dashboard-runner-card {{
            position: relative;
            overflow: hidden;
            border-radius: 22px;
            border: 1px solid rgba(20, 184, 166, 0.24);
            background:
                radial-gradient(circle at 12% 10%, rgba(45, 212, 191, 0.18), transparent 28%),
                linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(15, 118, 110, 0.88));
            padding: 1rem 1.1rem;
            color: #f8fafc;
            box-shadow: 0 18px 44px rgba(15, 23, 42, 0.16);
            margin: 0.45rem 0 1rem;
        }}
        .dashboard-runner-topline {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 0.7rem;
        }}
        .dashboard-runner-state {{
            display: flex;
            align-items: center;
            gap: 0.55rem;
            font-weight: 800;
            letter-spacing: -0.01em;
        }}
        .dashboard-runner-orb {{
            width: 0.72rem;
            height: 0.72rem;
            border-radius: 999px;
            background: #5eead4;
            box-shadow: 0 0 0 6px rgba(94, 234, 212, 0.15), 0 0 24px rgba(94, 234, 212, 0.95);
            animation: dashboardRunnerPulse 1.4s ease-in-out infinite;
        }}
        .dashboard-runner-percent {{
            font-variant-numeric: tabular-nums;
            color: rgba(248, 250, 252, 0.82);
            font-weight: 700;
        }}
        .dashboard-runner-track {{
            position: relative;
            height: 18px;
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.55);
            border: 1px solid rgba(255, 255, 255, 0.14);
            overflow: hidden;
        }}
        .dashboard-runner-bar {{
            position: absolute;
            inset: 0 auto 0 0;
            width: {shimmer_width}%;
            border-radius: 999px;
            background: linear-gradient(90deg, #14b8a6, #67e8f9, #facc15);
            box-shadow: 0 0 24px rgba(103, 232, 249, 0.35);
            transition: width 0.45s ease;
        }}
        .dashboard-runner-bar::after {{
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.55), transparent);
            animation: dashboardRunnerShimmer 1.6s linear infinite;
        }}
        .dashboard-runner-bar--done {{
            background: linear-gradient(90deg, #22c55e, #86efac);
        }}
        .dashboard-runner-bar--indeterminate {{
            width: 42%;
            animation: dashboardRunnerShimmer 1.8s ease-in-out infinite;
        }}
        .dashboard-runner-detail {{
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            margin-top: 0.65rem;
            color: rgba(248, 250, 252, 0.78);
            font-size: 0.86rem;
        }}
        .dashboard-runner-detail code {{
            color: #ccfbf1;
            background: rgba(15, 23, 42, 0.38);
            border-radius: 8px;
            padding: 0.1rem 0.35rem;
        }}
        </style>
        <div class="dashboard-runner-card">
            <div class="dashboard-runner-topline">
                <div class="dashboard-runner-state">
                    <span class="dashboard-runner-orb"></span>
                    <span>{safe_status}</span>
                </div>
                <div class="dashboard-runner-percent">{overall_percent}%</div>
            </div>
            <div class="dashboard-runner-track">
                <div class="{bar_class}"></div>
            </div>
            <div class="dashboard-runner-detail">
                <span><strong>Now:</strong> {safe_name}</span>
                <span><code>{safe_tqdm}</code></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    def _dashboard_runner_build_command(base_configs, model_files, models, batch_mode, batch_count, hist_mode, workers, graph_only):
        command = [
            sys.executable,
            RUNNER_SCRIPT,
            "--base-config",
            ",".join(base_configs),
            "--models-file",
            ",".join(model_files),
            "--models",
            ",".join(models),
        ]

        if graph_only:
            command.append("--graph-only")
        elif batch_mode == "Use config defaults":
            command.append("--default-batches")
        elif batch_mode == "All batches":
            command.extend(["--batches", "-1"])
        else:
            command.extend(["--batches", str(int(batch_count))])

        if hist_mode == "Enable histograms":
            command.append("--histograms")
        elif not graph_only:
            command.append("--no-histograms")

        if workers is not None:
            command.extend(["--workers", str(int(workers))])

        return command

    os.makedirs(RUNNER_OUTPUT_DIR, exist_ok=True)

    process = _dashboard_runner_current_process()
    is_running = process is not None
    last_log_path = st.session_state.get("dashboard_runner_log_path")
    last_command = st.session_state.get("dashboard_runner_command")
    last_returncode = st.session_state.get("dashboard_runner_returncode")
    log_text = _dashboard_runner_read_log(last_log_path)
    parsed_progress = _dashboard_runner_parse_progress(log_text)

    status_cols = st.columns(4)
    status_cols[0].metric("Runner Status", "Running" if is_running else "Idle")
    status_cols[1].metric("PID", str(process.pid) if is_running else "-")
    status_cols[2].metric("Last Exit", "-" if last_returncode is None else str(last_returncode))
    status_cols[3].metric("Log", os.path.basename(last_log_path) if last_log_path else "-")

    if last_command:
        with st.expander("Last command", expanded=False):
            st.code(last_command, language="bash")

    if last_log_path:
        st.markdown("#### Running Status")
        _dashboard_runner_render_loading(parsed_progress, is_running)
        total_configs = parsed_progress["total_configs"]
        current_config = parsed_progress["current_config"]
        if total_configs:
            config_ratio = current_config / total_configs if total_configs else 0.0
            config_label = f"Configs: {current_config}/{total_configs}"
            if parsed_progress["current_name"] and not parsed_progress["completed"]:
                config_label += f" - {parsed_progress['current_name']}"
            if parsed_progress["completed"]:
                config_label += " - completed"
            st.progress(min(1.0, max(0.0, config_ratio)), text=config_label)
        elif is_running:
            st.info("Run started. Waiting for configuration progress to appear in the log...")

        if parsed_progress["tqdm_percent"] is not None:
            tqdm_ratio = parsed_progress["tqdm_percent"] / 100.0
            tqdm_label = (
                f"Current step: {parsed_progress['tqdm_percent']}%"
                f" ({parsed_progress['tqdm_current']}/{parsed_progress['tqdm_total']})"
            )
            st.progress(min(1.0, max(0.0, tqdm_ratio)), text=tqdm_label)
            st.caption(parsed_progress["tqdm_line"])

    base_config_options = _dashboard_runner_list_files(BASE_CONFIGS_DIR)
    model_file_options = _dashboard_runner_list_files(INPUTS_DIR)

    if not base_config_options:
        st.error(f"No base config files found in `{BASE_CONFIGS_DIR}`.")
    elif not model_file_options:
        st.error(f"No model files found in `{INPUTS_DIR}`.")
    else:
        default_base = ["advanced_full_config_fp8e4m3.yaml"] if "advanced_full_config_fp8e4m3.yaml" in base_config_options else [base_config_options[0]]
        default_model_files = ["models_best_acc1.yaml"] if "models_best_acc1.yaml" in model_file_options else [model_file_options[0]]
        generated_config_data = None
        generated_config_name = None
        generated_config_error = None

        config_source = st.radio(
            "Configuration source",
            options=["Use existing base config", "Build config in dashboard"],
            index=0,
            horizontal=True,
            key="runner_config_source",
        )
        selected_base_configs = []

        if config_source == "Use existing base config":
            selected_base_configs = st.multiselect(
                "Base config",
                options=base_config_options,
                default=default_base,
                key="runner_base_configs",
                help="Files from runspace/inputs/base_configs/.",
            )
        else:
            st.markdown("#### Build Configuration")
            builder_top1, builder_top2 = st.columns([1, 1])
            template_config_name = builder_top1.selectbox(
                "Start from template",
                options=base_config_options,
                index=base_config_options.index(default_base[0]) if default_base[0] in base_config_options else 0,
                key="runner_builder_template",
            )
            template_key = _dashboard_runner_safe_name(template_config_name)
            def _builder_key(name):
                return f"runner_builder_{template_key}_{name}"

            config_name = builder_top2.text_input(
                "Generated config name",
                value=f"dashboard_{datetime.now().strftime('%Y%m%d')}",
                key=_builder_key("config_name"),
                help="Saved into runspace/inputs/base_configs/ when you save or run.",
            )
            template_path = os.path.join(BASE_CONFIGS_DIR, template_config_name)
            try:
                template_config = _dashboard_runner_load_yaml(template_path)
            except Exception as exc:
                template_config = {}
                generated_config_error = f"Could not load template `{template_config_name}`: {exc}"

            template_adapter = template_config.get("adapter", {}) if isinstance(template_config, dict) else {}
            template_quant = template_config.get("quantization", {}) if isinstance(template_config, dict) else {}
            template_dataset = template_config.get("dataset", {}) if isinstance(template_config, dict) else {}
            template_eval = template_config.get("evaluation", {}) if isinstance(template_config, dict) else {}

            with st.expander("Quantization and adapter", expanded=True):
                qa1, qa2, qa3 = st.columns(3)
                weight_quantization = qa1.checkbox(
                    "Weight quantization",
                    value=bool(template_adapter.get("weight_quantization", False)),
                    key=_builder_key("weight_quant"),
                )
                input_quantization = qa2.checkbox(
                    "Input quantization",
                    value=bool(template_adapter.get("input_quantization", True)),
                    key=_builder_key("input_quant"),
                )
                quantize_first_layer = qa3.checkbox(
                    "Quantize first layer",
                    value=bool(template_adapter.get("quantize_first_layer", False)),
                    key=_builder_key("first_layer"),
                )

                q1, q2, q3 = st.columns(3)
                quant_formats = [
                    "fp8_e4m3", "fp8_e5m2", "fp8_e1m6", "fp6_e2m3",
                    "fp4_e2m1", "fp4_e1m2", "int8", "int4", "fp32",
                ]
                template_format = str(template_quant.get("format", "fp8_e4m3"))
                weight_format = q1.selectbox(
                    "Weight format",
                    options=quant_formats,
                    index=quant_formats.index(template_format) if template_format in quant_formats else 0,
                    key=_builder_key("weight_format"),
                )
                template_input_format = str(template_quant.get("input_format", template_format))
                input_format = q2.selectbox(
                    "Input format",
                    options=quant_formats,
                    index=quant_formats.index(template_input_format) if template_input_format in quant_formats else 0,
                    key=_builder_key("input_format"),
                )
                calib_method = q3.text_input(
                    "Calibration method",
                    value=str(template_quant.get("calib_method", "max")),
                    key=_builder_key("calib"),
                )

                m1, m2, m3, m4 = st.columns(4)
                modes = ["tensor", "channel", "chunk"]
                template_input_mode = str(template_quant.get("mode", "chunk"))
                input_mode = m1.selectbox(
                    "Input mode",
                    options=modes,
                    index=modes.index(template_input_mode) if template_input_mode in modes else 2,
                    key=_builder_key("input_mode"),
                )
                input_chunk_size = m2.number_input(
                    "Input chunk size",
                    min_value=1,
                    max_value=100000,
                    value=int(template_quant.get("chunk_size", 128) or 128),
                    step=1,
                    key=_builder_key("input_chunk"),
                )
                template_weight_mode = str(template_quant.get("weight_mode", "tensor"))
                weight_mode = m3.selectbox(
                    "Weight mode",
                    options=modes,
                    index=modes.index(template_weight_mode) if template_weight_mode in modes else 0,
                    key=_builder_key("weight_mode"),
                )
                weight_chunk_size = m4.number_input(
                    "Weight chunk size",
                    min_value=1,
                    max_value=100000,
                    value=int(template_quant.get("weight_chunk_size", 128) or 128),
                    step=1,
                    key=_builder_key("weight_chunk"),
                )

                ops1, ops2, ops3 = st.columns([2, 2, 1])
                quantized_ops = ops1.text_input(
                    "Quantized ops",
                    value=_dashboard_runner_csv_from_list(template_adapter.get("quantized_ops"), ["all"]),
                    key=_builder_key("quantized_ops"),
                    help="Comma-separated, for example: all or Conv2d,Linear.",
                )
                excluded_ops = ops2.text_input(
                    "Excluded ops",
                    value=_dashboard_runner_csv_from_list(template_adapter.get("excluded_ops"), [""]),
                    key=_builder_key("excluded_ops"),
                )
                rounding = ops3.selectbox(
                    "Rounding",
                    options=["nearest", "truncate"],
                    index=0 if str(template_quant.get("rounding", "nearest")) == "nearest" else 1,
                    key=_builder_key("rounding"),
                )

            with st.expander("Dataset and evaluation", expanded=True):
                d1, d2 = st.columns([1, 2])
                dataset_name = d1.text_input(
                    "Dataset name",
                    value=str(template_dataset.get("name", "imagenet")),
                    key=_builder_key("dataset_name"),
                )
                dataset_path = d2.text_input(
                    "Dataset path",
                    value=str(template_dataset.get("path", "/data/imagenet/val")),
                    key=_builder_key("dataset_path"),
                )
                d3, d4, d5 = st.columns(3)
                batch_size = d3.number_input(
                    "Dataset batch size",
                    min_value=1,
                    max_value=4096,
                    value=int(template_dataset.get("batch_size", 128) or 128),
                    step=1,
                    key=_builder_key("batch_size"),
                )
                num_workers = d4.number_input(
                    "Dataset workers",
                    min_value=0,
                    max_value=256,
                    value=int(template_dataset.get("num_workers", 32) or 32),
                    step=1,
                    key=_builder_key("num_workers"),
                )
                compare_batches = d5.number_input(
                    "Default compare batches",
                    min_value=-1,
                    max_value=100000,
                    value=int(template_eval.get("compare_batches", -1) or -1),
                    step=1,
                    key=_builder_key("compare_batches"),
                    help="-1 means all batches.",
                )

                e1, e2, e3 = st.columns(3)
                evaluation_mode = e1.selectbox(
                    "Evaluation mode",
                    options=["compare", "evaluate"],
                    index=0 if str(template_eval.get("mode", "compare")) == "compare" else 1,
                    key=_builder_key("eval_mode"),
                )
                generate_graph_svg = e2.checkbox(
                    "Generate graph SVG",
                    value=bool(template_eval.get("generate_graph_svg", False)),
                    key=_builder_key("graph_svg"),
                )
                save_histograms = e3.checkbox(
                    "Save histograms by default",
                    value=bool(template_eval.get("save_histograms", False)),
                    key=_builder_key("save_hist"),
                )

            form_values = {
                "config_name": config_name,
                "template_config": template_config_name,
                "quantize_first_layer": quantize_first_layer,
                "weight_quantization": weight_quantization,
                "input_quantization": input_quantization,
                "quantized_ops": quantized_ops,
                "excluded_ops": excluded_ops,
                "weight_format": weight_format,
                "input_format": input_format,
                "input_mode": input_mode,
                "input_chunk_size": input_chunk_size,
                "weight_mode": weight_mode,
                "weight_chunk_size": weight_chunk_size,
                "calib_method": calib_method,
                "rounding": rounding,
                "dataset_name": dataset_name,
                "dataset_path": dataset_path,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "evaluation_mode": evaluation_mode,
                "compare_batches": compare_batches,
                "generate_graph_svg": generate_graph_svg,
                "save_histograms": save_histograms,
            }
            generated_config_data = _dashboard_runner_build_config_from_form(template_config, form_values)
            generated_yaml = _dashboard_runner_dump_yaml(generated_config_data)
            yaml_editor_key = _builder_key("yaml_editor")
            yaml_auto_key = _builder_key("yaml_last_auto")
            yaml_is_manual = (
                yaml_editor_key in st.session_state
                and yaml_auto_key in st.session_state
                and st.session_state[yaml_editor_key] != st.session_state[yaml_auto_key]
            )
            reset_yaml_col, yaml_state_col = st.columns([1, 3])
            if reset_yaml_col.button("Reset YAML From Form", key=_builder_key("yaml_reset"), width='stretch'):
                st.session_state[yaml_editor_key] = generated_yaml
                st.session_state[yaml_auto_key] = generated_yaml
                st.rerun()
            if not yaml_is_manual:
                st.session_state[yaml_editor_key] = generated_yaml
            st.session_state[yaml_auto_key] = generated_yaml
            if yaml_is_manual:
                yaml_state_col.caption("Advanced YAML edits are active. Form changes will not overwrite the editor until you reset it.")
            else:
                yaml_state_col.caption("Live preview is synced with the form controls.")
            edited_yaml = st.text_area(
                "Generated YAML preview / advanced edit",
                height=360,
                key=yaml_editor_key,
                help="You can edit the YAML before saving or running.",
            )
            try:
                edited_config = yaml.safe_load(edited_yaml) or {}
                if not isinstance(edited_config, dict):
                    raise ValueError("Generated YAML must be a mapping/object.")
                generated_config_data = edited_config
            except Exception as exc:
                generated_config_error = f"Generated YAML is invalid: {exc}"

            generated_config_name = _dashboard_runner_safe_name(config_name)
            selected_base_configs = [f"{generated_config_name}.yaml"]
            save_col, path_col = st.columns([1, 3])
            if generated_config_error:
                st.error(generated_config_error)
            if save_col.button(
                "Save Generated Config",
                key=_builder_key("save_generated_config"),
                width='stretch',
                disabled=bool(generated_config_error),
            ):
                saved_name, saved_path = _dashboard_runner_save_generated_config(generated_config_name, generated_config_data)
                st.success(f"Saved `{saved_name}` to `{saved_path}`.")
            path_col.caption(f"Will run as `--base-config {selected_base_configs[0]}`.")

        col_models = st.container()
        selected_model_files = col_models.multiselect(
            "Models file",
            options=model_file_options,
            default=default_model_files,
            key="runner_model_files",
            help="Model-list YAML files from runspace/inputs/.",
        )

        available_models = _dashboard_runner_load_models(selected_model_files)
        default_models = ["resnet18"] if "resnet18" in available_models else available_models[:1]
        selected_models = st.multiselect(
            "Models",
            options=available_models,
            default=default_models,
            key="runner_models",
            help="Pick one or more models to run. This maps to --models.",
        )

        col_batch, col_hist, col_workers = st.columns(3)
        graph_only = col_batch.checkbox(
            "Graph only",
            value=False,
            key="runner_graph_only",
            help="Generate quantization graphs and skip evaluation.",
        )
        batch_mode = col_batch.radio(
            "Batches",
            options=["Use config defaults", "Custom batch count", "All batches"],
            index=1,
            key="runner_batch_mode",
            disabled=graph_only,
        )
        batch_count = col_batch.number_input(
            "Batch count",
            min_value=1,
            max_value=100000,
            value=5,
            step=1,
            key="runner_batch_count",
            disabled=graph_only or batch_mode != "Custom batch count",
        )

        hist_mode = col_hist.radio(
            "Histograms",
            options=["Config defaults", "Enable histograms"],
            index=0,
            key="runner_hist_mode",
            disabled=graph_only,
            help="Config defaults maps to --no-histograms, matching the current interactive runner behavior.",
        )

        override_workers = col_workers.checkbox(
            "Override workers",
            value=False,
            key="runner_override_workers",
        )
        workers = col_workers.number_input(
            "dataset.num_workers",
            min_value=0,
            max_value=256,
            value=8,
            step=1,
            key="runner_workers",
            disabled=not override_workers,
        )

        can_build = bool(selected_base_configs and selected_model_files and selected_models and not generated_config_error)
        if can_build:
            command = _dashboard_runner_build_command(
                selected_base_configs,
                selected_model_files,
                selected_models,
                batch_mode,
                batch_count,
                hist_mode,
                int(workers) if override_workers else None,
                graph_only,
            )
            command_preview = shlex.join(command)
        else:
            command = []
            command_preview = ""

        st.markdown("#### Command Preview")
        if command_preview:
            st.code(command_preview, language="bash")
        else:
            st.info("Choose at least one base config, model file, and model to enable the run button.")

        run_col, refresh_col = st.columns([1, 1])
        if run_col.button(
            "🚀 Run Selected Models",
            key="runner_start_btn",
            type="primary",
            width='stretch',
            disabled=is_running or not can_build,
        ):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(RUNNER_OUTPUT_DIR, f"dashboard_run_{timestamp}.log")
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            if config_source == "Build config in dashboard":
                saved_name, saved_path = _dashboard_runner_save_generated_config(generated_config_name, generated_config_data)
                selected_base_configs = [saved_name]
                command = _dashboard_runner_build_command(
                    selected_base_configs,
                    selected_model_files,
                    selected_models,
                    batch_mode,
                    batch_count,
                    hist_mode,
                    int(workers) if override_workers else None,
                    graph_only,
                )
                command_preview = shlex.join(command)
            with open(log_path, "ab", buffering=0) as log_file:
                log_file.write((f"$ {command_preview}\n\n").encode("utf-8"))
                if config_source == "Build config in dashboard":
                    log_file.write((f"# Saved dashboard config: {saved_path}\n\n").encode("utf-8"))
                started_process = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
            st.session_state["dashboard_runner_process"] = started_process
            st.session_state["dashboard_runner_command"] = command_preview
            st.session_state["dashboard_runner_log_path"] = log_path
            st.session_state["dashboard_runner_returncode"] = None
            st.success(f"Started run PID {started_process.pid}.")
            time.sleep(0.2)
            st.rerun()

        if refresh_col.button("🔄 Refresh Runner Status", key="runner_refresh_btn", width='stretch'):
            st.rerun()

    st.markdown("---")
    st.markdown("#### Run Log")
    if last_log_path:
        st.caption(f"`{last_log_path}`")
        log_lines = _dashboard_runner_lines(log_text)
        auto_refresh = st.checkbox(
            "Auto-refresh",
            value=True,
            key="runner_auto_refresh",
            disabled=not is_running,
        )
        page_size = st.select_slider(
            "Lines per page",
            options=[250, 500, 1000, 2000, 5000],
            value=1000,
            key="runner_log_page_size",
            help="Paged output avoids browser/widget truncation on very large logs.",
        )
        page_count = max(1, (len(log_lines) + page_size - 1) // page_size)
        st.session_state["runner_log_page_number"] = min(
            int(st.session_state.get("runner_log_page_number", page_count)),
            page_count,
        )
        nav_top, nav_prev, nav_page, nav_next, nav_bottom = st.columns([1, 1, 2, 1, 1])
        if nav_top.button("Top", key="runner_log_top", width='stretch'):
            st.session_state["runner_log_page_number"] = 1
            st.rerun()
        if nav_prev.button("Prev", key="runner_log_prev", width='stretch', disabled=st.session_state["runner_log_page_number"] <= 1):
            st.session_state["runner_log_page_number"] -= 1
            st.rerun()
        page_number = nav_page.number_input(
            "Page",
            min_value=1,
            max_value=page_count,
            value=st.session_state["runner_log_page_number"],
            step=1,
        )
        st.session_state["runner_log_page_number"] = int(page_number)
        if nav_next.button("Next", key="runner_log_next", width='stretch', disabled=page_number >= page_count):
            st.session_state["runner_log_page_number"] += 1
            st.rerun()
        if nav_bottom.button("Bottom", key="runner_log_bottom", width='stretch'):
            st.session_state["runner_log_page_number"] = page_count
            st.rerun()

        start_idx = (int(page_number) - 1) * page_size
        end_idx = min(len(log_lines), start_idx + page_size)
        shown_log = "\n".join(log_lines[start_idx:end_idx])
        st.caption(
            f"Showing lines {start_idx + 1 if log_lines else 0}-{end_idx} "
            f"of {len(log_lines)}. Page {int(page_number)}/{page_count}."
        )

        st.text_area(
            "Runner output",
            value=shown_log,
            height=720,
            disabled=True,
            label_visibility="collapsed",
        )
        st.download_button(
            "Download full log",
            data=log_text,
            file_name=os.path.basename(last_log_path),
            mime="text/plain",
            key="runner_download_log",
            width='stretch',
        )
    else:
        st.info("No dashboard-started run yet.")

    if is_running and st.session_state.get("runner_auto_refresh", True):
        time.sleep(2)
        st.rerun()
