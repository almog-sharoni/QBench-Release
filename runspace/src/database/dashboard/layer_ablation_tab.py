"""Data preparation and rendering for weight layer-type ablation runs."""

import json

import pandas as pd
import streamlit as st


LAYER_ABLATION_STRATEGY_ORDER = [
    "Uniform",
    "Layer optimal",
    "Chunk optimal",
]

LAYER_ABLATION_PLOT_COLUMNS = [
    "Model",
    "Layer Type",
    "Bits",
    "Strategy",
    "Series",
    "Weight Format",
    "Optimization Metric",
    "Accuracy (%)",
    "Reference Accuracy (%)",
    "Accuracy Drop (pp)",
    "Target Layer Count",
    "Evaluation Setup",
    "Evaluation Scope",
    "Model Weights",
    "Dataset",
    "Batch Size",
    "MSE",
    "L1",
    "Experiment Type",
    "Run Date",
    "Run ID",
]


def _layer_ablation_text(value, default=""):
    if value is None:
        return default
    try:
        if bool(pd.isna(value)):
            return default
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text else default


def _layer_ablation_json(value):
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    try:
        if bool(pd.isna(value)):
            return {}
    except (TypeError, ValueError):
        pass
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _layer_ablation_positive_int(value):
    number = _layer_ablation_int(value)
    return number if number is not None and number > 0 else None


def _layer_ablation_int(value):
    try:
        if value is None or bool(pd.isna(value)):
            return None
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    return number


def _layer_ablation_cli_int(cli_command, option_name):
    tokens = _layer_ablation_text(cli_command).split()
    option = f"--{option_name}"
    for index, token in enumerate(tokens):
        if token == option and index + 1 < len(tokens):
            return _layer_ablation_int(tokens[index + 1])
        if token.startswith(f"{option}="):
            return _layer_ablation_int(token.split("=", 1)[1])
    return None


def _layer_ablation_experiment_parts(experiment_type):
    """Return the fallback layer slug and width encoded in an ablation run type."""
    normalized = _layer_ablation_text(experiment_type).lower().replace("-", "_")
    marker = "_ablation_"
    if marker not in normalized:
        return None, None
    suffix = normalized.split(marker, 1)[1]
    layer_slug, separator, bit_text = suffix.rpartition("_")
    if not separator or not layer_slug or not bit_text.isdigit():
        return None, None
    return layer_slug, _layer_ablation_positive_int(bit_text)


def _layer_ablation_config_candidates(raw_config):
    config = _layer_ablation_json(raw_config)
    if not config:
        return []

    candidates = [config]
    experiment = config.get("experiment")
    if isinstance(experiment, dict):
        nested_config = _layer_ablation_json(experiment.get("config_json"))
        if nested_config:
            candidates.append(nested_config)
    return candidates


def _layer_ablation_metadata(row):
    experiment_type = _layer_ablation_text(row.get("experiment_type"))
    fallback_layer_type, fallback_bits = _layer_ablation_experiment_parts(
        experiment_type
    )

    layer_type = None
    bit_width = None
    target_layer_count = None
    has_target_only_metadata = False
    model_weights = ""
    dataset_name = ""
    dataset_path = ""
    batch_size = None
    max_batches = None

    for config in _layer_ablation_config_candidates(row.get("config_json")):
        quantization = config.get("quantization")
        quantization = quantization if isinstance(quantization, dict) else {}
        experiment = config.get("experiment")
        experiment = experiment if isinstance(experiment, dict) else {}
        model = config.get("model")
        model = model if isinstance(model, dict) else {}
        dataset = config.get("dataset")
        dataset = dataset if isinstance(dataset, dict) else {}
        evaluation = config.get("evaluation")
        evaluation = evaluation if isinstance(evaluation, dict) else {}

        if not layer_type:
            layer_type = _layer_ablation_text(
                quantization.get("target_layer_type")
                or experiment.get("ablation_layer_type")
                or config.get("ablation_layer_type")
            ) or None
        if bit_width is None:
            bit_width = _layer_ablation_positive_int(
                quantization.get("optimization_bit_width")
                or experiment.get("bit_width")
                or config.get("optimization_bit_width")
            )
        if target_layer_count is None:
            target_layer_count = _layer_ablation_positive_int(
                quantization.get("target_layer_count")
                or experiment.get("target_layer_count")
                or config.get("target_layer_count")
            )

        policy = _layer_ablation_text(
            experiment.get("ablation_policy") or config.get("ablation_policy")
        ).lower()
        non_target_format = _layer_ablation_text(
            quantization.get("non_target_weight_format")
            or experiment.get("non_target_format")
            or config.get("non_target_weight_format")
        ).lower()
        has_target_only_metadata = has_target_only_metadata or (
            policy == "target_only" or non_target_format == "fp32"
        )
        if not model_weights:
            model_weights = _layer_ablation_text(model.get("weights"))
        if not dataset_name:
            dataset_name = _layer_ablation_text(dataset.get("name"))
        if not dataset_path:
            dataset_path = _layer_ablation_text(dataset.get("path"))
        if batch_size is None:
            batch_size = _layer_ablation_positive_int(dataset.get("batch_size"))
        if max_batches is None:
            max_batches = _layer_ablation_int(
                evaluation.get("max_batches")
                if evaluation.get("max_batches") is not None
                else evaluation.get("limit_batches")
            )
        if max_batches is None:
            max_batches = _layer_ablation_int(dataset.get("limit_batches"))

    layer_type = layer_type or fallback_layer_type
    bit_width = bit_width or fallback_bits
    fallback_is_weight_ablation = (
        fallback_layer_type is not None
        and experiment_type.lower().replace("-", "_").startswith("weight_quant_")
    )
    is_ablation = fallback_is_weight_ablation or (
        bool(layer_type) and has_target_only_metadata
    )
    if not is_ablation or not layer_type or bit_width is None:
        return None

    if max_batches is None:
        max_batches = _layer_ablation_cli_int(
            row.get("cli_command"), "limit_batches"
        )
    if max_batches is None:
        evaluation_scope = "Unknown scope"
    elif max_batches <= 0:
        evaluation_scope = "Full dataset"
    else:
        evaluation_scope = (
            f"{max_batches} batch{'es' if max_batches != 1 else ''}"
        )
    if dataset_name and dataset_path:
        dataset_label = f"{dataset_name} [{dataset_path}]"
    else:
        dataset_label = dataset_name or dataset_path or "Unknown dataset"
    weights_label = model_weights or "Unknown weights"
    setup_parts = [weights_label, dataset_label, evaluation_scope]
    if batch_size is not None:
        setup_parts.append(f"batch size {batch_size}")
    evaluation_setup = " · ".join(setup_parts)

    return {
        "layer_type": layer_type,
        "bit_width": bit_width,
        "target_layer_count": target_layer_count,
        "evaluation_setup": evaluation_setup,
        "evaluation_scope": evaluation_scope,
        "model_weights": weights_label,
        "dataset": dataset_label,
        "batch_size": batch_size,
    }


def _layer_ablation_strategy(weight_format):
    normalized = _layer_ablation_text(weight_format).lower().replace("-", "_")
    if normalized.startswith("opt_layer_") or normalized.startswith(
        "optimized_layer_"
    ):
        return "Layer optimal"
    if normalized.startswith("opt_chunk_") or normalized.startswith(
        "optimized_chunk_"
    ):
        return "Chunk optimal"
    return "Uniform"


def _layer_ablation_metric(weight_format, strategy):
    if strategy == "Uniform":
        return ""
    normalized = _layer_ablation_text(weight_format).lower().replace("-", "_")
    for prefix in (
        "opt_layer_",
        "optimized_layer_",
        "opt_chunk_",
        "optimized_chunk_",
    ):
        if normalized.startswith(prefix):
            return normalized[len(prefix):].upper()
    return ""


def _layer_ablation_reference_map(df):
    if df is None or df.empty or "model_name" not in df.columns:
        return {}

    weight_dt = df.get(
        "weight_dt", pd.Series("", index=df.index, dtype=object)
    ).fillna("").astype(str).str.lower()
    activation_dt = df.get(
        "activation_dt", pd.Series("", index=df.index, dtype=object)
    ).fillna("").astype(str).str.lower()
    accuracy = pd.to_numeric(
        df.get("acc1", pd.Series(float("nan"), index=df.index)),
        errors="coerce",
    )
    references = df[
        weight_dt.eq("fp32") & activation_dt.eq("fp32") & accuracy.gt(0)
    ].copy()
    if references.empty:
        return {}

    reference_status = references.get(
        "status", pd.Series("", index=references.index, dtype=object)
    ).fillna("").astype(str).str.upper()
    references = references[reference_status.isin(["", "SUCCESS"])]
    if references.empty:
        return {}

    references["_ablation_ref_accuracy"] = pd.to_numeric(
        references["acc1"], errors="coerce"
    )
    references["_ablation_is_explicit_ref"] = references.get(
        "experiment_type", pd.Series("", index=references.index, dtype=object)
    ).fillna("").astype(str).str.lower().eq("fp32_ref").astype(int)
    references["_ablation_is_success"] = references.get(
        "status", pd.Series("", index=references.index, dtype=object)
    ).fillna("").astype(str).str.upper().eq("SUCCESS").astype(int)
    references["_ablation_run_date"] = pd.to_datetime(
        references.get(
            "run_date", pd.Series(None, index=references.index, dtype=object)
        ),
        errors="coerce",
    )
    references["_ablation_run_id"] = pd.to_numeric(
        references.get("id", pd.Series(-1, index=references.index)),
        errors="coerce",
    ).fillna(-1)
    references = references.sort_values(
        by=[
            "model_name",
            "_ablation_is_explicit_ref",
            "_ablation_is_success",
            "_ablation_run_date",
            "_ablation_run_id",
        ],
        ascending=[True, False, False, False, False],
    ).drop_duplicates(subset=["model_name"], keep="first")
    return references.set_index("model_name")["_ablation_ref_accuracy"].to_dict()


def build_layer_ablation_plot_df(df):
    """Build chart-ready rows for target-only weighted layer-type ablations."""
    if df is None or df.empty:
        return pd.DataFrame(columns=LAYER_ABLATION_PLOT_COLUMNS)

    working_df = df.copy()
    if "status" in working_df.columns:
        status = working_df["status"].fillna("").astype(str).str.upper()
        working_df = working_df[status.isin(["", "SUCCESS"])]
    if working_df.empty:
        return pd.DataFrame(columns=LAYER_ABLATION_PLOT_COLUMNS)

    model_reference = _layer_ablation_reference_map(df)
    rows = []
    for _, run_row in working_df.iterrows():
        metadata = _layer_ablation_metadata(run_row)
        if metadata is None:
            continue

        activation_format = _layer_ablation_text(
            run_row.get("activation_dt")
        ).lower()
        weight_format = _layer_ablation_text(run_row.get("weight_dt"))
        if activation_format != "fp32" or weight_format.lower() in {"", "fp32"}:
            continue

        accuracy = pd.to_numeric(
            pd.Series([run_row.get("acc1")]), errors="coerce"
        ).iloc[0]
        if pd.isna(accuracy):
            continue

        model = _layer_ablation_text(run_row.get("model_name"), "unknown")
        reference_accuracy = pd.to_numeric(
            pd.Series([run_row.get("ref_acc1")]), errors="coerce"
        ).iloc[0]
        if pd.isna(reference_accuracy) or reference_accuracy <= 0:
            reference_accuracy = model_reference.get(model, float("nan"))

        strategy = _layer_ablation_strategy(weight_format)
        layer_type = metadata["layer_type"]
        rows.append(
            {
                "Model": model,
                "Layer Type": layer_type,
                "Bits": int(metadata["bit_width"]),
                "Strategy": strategy,
                "Series": f"{layer_type} · {strategy}",
                "Weight Format": weight_format,
                "Optimization Metric": _layer_ablation_metric(
                    weight_format, strategy
                ),
                "Accuracy (%)": float(accuracy),
                "Reference Accuracy (%)": (
                    float(reference_accuracy)
                    if not pd.isna(reference_accuracy)
                    else float("nan")
                ),
                "Accuracy Drop (pp)": (
                    float(reference_accuracy - accuracy)
                    if not pd.isna(reference_accuracy)
                    else float("nan")
                ),
                "Target Layer Count": metadata["target_layer_count"],
                "Evaluation Setup": metadata["evaluation_setup"],
                "Evaluation Scope": metadata["evaluation_scope"],
                "Model Weights": metadata["model_weights"],
                "Dataset": metadata["dataset"],
                "Batch Size": metadata["batch_size"],
                "MSE": pd.to_numeric(
                    pd.Series([run_row.get("mse")]), errors="coerce"
                ).iloc[0],
                "L1": pd.to_numeric(
                    pd.Series([run_row.get("l1")]), errors="coerce"
                ).iloc[0],
                "Experiment Type": _layer_ablation_text(
                    run_row.get("experiment_type")
                ),
                "Run Date": _layer_ablation_text(run_row.get("run_date")),
                "Run ID": run_row.get("id"),
            }
        )

    plot_df = pd.DataFrame(rows, columns=LAYER_ABLATION_PLOT_COLUMNS)
    if plot_df.empty:
        return plot_df

    plot_df["Strategy"] = pd.Categorical(
        plot_df["Strategy"],
        categories=LAYER_ABLATION_STRATEGY_ORDER,
        ordered=True,
    )
    plot_df["_ablation_run_date"] = pd.to_datetime(
        plot_df["Run Date"], errors="coerce"
    )
    plot_df["_ablation_run_id"] = pd.to_numeric(
        plot_df["Run ID"], errors="coerce"
    ).fillna(-1)
    plot_df = plot_df.sort_values(
        by=[
            "Model",
            "Layer Type",
            "Bits",
            "Strategy",
            "Weight Format",
            "_ablation_run_date",
            "_ablation_run_id",
        ],
        ascending=[True, True, True, True, True, False, False],
    )
    return plot_df.drop(columns=["_ablation_run_date", "_ablation_run_id"])


def _layer_ablation_best_per_strategy(plot_df):
    if plot_df is None or plot_df.empty:
        return pd.DataFrame(columns=LAYER_ABLATION_PLOT_COLUMNS)
    accuracy = pd.to_numeric(plot_df["Accuracy (%)"], errors="coerce")
    candidates = plot_df[accuracy.notna()].copy()
    if candidates.empty:
        return candidates
    best_indices = candidates.groupby(
        ["Model", "Evaluation Setup", "Layer Type", "Bits", "Strategy"],
        observed=True,
        dropna=False,
    )["Accuracy (%)"].idxmax()
    return candidates.loc[best_indices].sort_values(
        ["Layer Type", "Strategy", "Bits"]
    )


def _render_layer_ablation_dashboard(source_df):
    plot_df = build_layer_ablation_plot_df(source_df)
    if plot_df.empty:
        st.info(
            "No target-only weight layer ablation runs are available for this slice. "
            "Run `find_optimal_weight_quant.py --ablation_layer_types ...` to populate it."
        )
        return

    st.markdown(
        """
        <div class="dashboard-section-title">
            <div>
                <h3>Weighted Layer-Type Ablations</h3>
                <p>Compare uniform formats with layer- and chunk-optimal choices while every non-target weight remains FP32.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    available_models = sorted(plot_df["Model"].dropna().astype(str).unique())
    model_key = "layer_ablation_model"
    if st.session_state.get(model_key) not in available_models:
        st.session_state[model_key] = available_models[0]

    model_col, setup_col, view_col, metric_col = st.columns([2, 2, 1, 1])
    selected_model = model_col.selectbox(
        "Model",
        options=available_models,
        key=model_key,
    )
    model_df = plot_df[plot_df["Model"].eq(selected_model)].copy()
    available_setups = model_df["Evaluation Setup"].value_counts().index.tolist()
    setup_key = "layer_ablation_evaluation_setup"
    if st.session_state.get(setup_key) not in available_setups:
        st.session_state[setup_key] = available_setups[0]
    selected_setup = setup_col.selectbox(
        "Evaluation setup",
        options=available_setups,
        key=setup_key,
        help="Runs are separated by model weights, dataset, and evaluation scope before best formats are selected.",
    )
    model_df = model_df[model_df["Evaluation Setup"].eq(selected_setup)].copy()
    view_mode = view_col.selectbox(
        "View",
        options=["Best per strategy", "All runs"],
        key="layer_ablation_view_mode",
        help="Best per strategy keeps the most accurate uniform format and each optimal policy at every width.",
    )
    metric_label = metric_col.selectbox(
        "Metric",
        options=["Top-1 accuracy", "Top-1 accuracy drop"],
        key="layer_ablation_metric",
    )

    available_layer_types = sorted(
        model_df["Layer Type"].dropna().astype(str).unique()
    )
    layer_key = "layer_ablation_layer_types"
    if layer_key in st.session_state:
        selected_state = [
            layer_type
            for layer_type in st.session_state[layer_key]
            if layer_type in available_layer_types
        ]
        st.session_state[layer_key] = selected_state or available_layer_types
    selected_layer_types = st.multiselect(
        "Layer types",
        options=available_layer_types,
        default=available_layer_types,
        key=layer_key,
        help="These are concrete weight-owning module types, including modules nested inside transformers.",
    )
    if not selected_layer_types:
        st.info("Select at least one layer type.")
        return

    strategy_options = [
        strategy
        for strategy in LAYER_ABLATION_STRATEGY_ORDER
        if strategy in set(model_df["Strategy"].dropna().astype(str))
    ]
    strategy_key = "layer_ablation_strategies"
    if strategy_key in st.session_state:
        selected_state = [
            strategy
            for strategy in st.session_state[strategy_key]
            if strategy in strategy_options
        ]
        st.session_state[strategy_key] = selected_state or strategy_options
    selected_strategies = st.multiselect(
        "Strategies",
        options=strategy_options,
        default=strategy_options,
        key=strategy_key,
    )
    if not selected_strategies:
        st.info("Select at least one strategy.")
        return

    filtered_df = model_df[
        model_df["Layer Type"].isin(selected_layer_types)
        & model_df["Strategy"].astype(str).isin(selected_strategies)
    ].copy()
    if filtered_df.empty:
        st.info("No ablation rows match the selected filters.")
        return

    best_df = _layer_ablation_best_per_strategy(filtered_df)
    chart_df = best_df.copy() if view_mode == "Best per strategy" else filtered_df
    y_field = (
        "Accuracy (%)"
        if metric_label == "Top-1 accuracy"
        else "Accuracy Drop (pp)"
    )
    y_title = (
        "Top-1 Accuracy (%)"
        if metric_label == "Top-1 accuracy"
        else "Top-1 Accuracy Drop (percentage points)"
    )
    ref_values = pd.to_numeric(
        filtered_df["Reference Accuracy (%)"], errors="coerce"
    ).dropna()
    unique_ref_values = sorted(ref_values.round(8).unique().tolist())
    mixed_references = len(unique_ref_values) > 1
    if metric_label == "Top-1 accuracy" and len(unique_ref_values) == 1:
        chart_df["Reference Line"] = unique_ref_values[0]
    elif metric_label == "Top-1 accuracy":
        chart_df["Reference Line"] = float("nan")
    else:
        chart_df["Reference Line"] = 0.0

    if mixed_references:
        st.warning(
            "These rows contain multiple FP32 reference accuracies. Each point's "
            "drop uses its own logged reference; the absolute-accuracy reference "
            "line is omitted."
        )
    best_accuracy = pd.to_numeric(
        filtered_df["Accuracy (%)"], errors="coerce"
    ).max()
    metric_columns = st.columns(4)
    metric_columns[0].metric("Runs", f"{len(filtered_df):,}")
    metric_columns[1].metric("Layer types", len(selected_layer_types))
    metric_columns[2].metric(
        "FP32 reference",
        (
            "Mixed"
            if mixed_references
            else f"{unique_ref_values[0]:.3f}%"
            if unique_ref_values
            else "Unknown"
        ),
    )
    metric_columns[3].metric(
        "Best ablation",
        f"{best_accuracy:.3f}%" if pd.notna(best_accuracy) else "Unknown",
    )

    shared_encoding = {
        "x": {
            "field": "Bits",
            "type": "quantitative",
            "axis": {"title": "Target Weight Bit Width", "tickMinStep": 1},
        },
        "y": {
            "field": y_field,
            "type": "quantitative",
            "scale": {"zero": False, "nice": True, "padding": 10},
            "axis": {"title": y_title},
        },
    }
    tooltip = [
        {"field": "Model", "type": "nominal"},
        {"field": "Evaluation Scope", "type": "nominal"},
        {"field": "Model Weights", "type": "nominal"},
        {"field": "Dataset", "type": "nominal"},
        {"field": "Batch Size", "type": "quantitative", "format": "d"},
        {"field": "Layer Type", "type": "nominal"},
        {"field": "Strategy", "type": "nominal"},
        {"field": "Bits", "type": "quantitative", "format": "d"},
        {"field": "Weight Format", "type": "nominal"},
        {"field": "Optimization Metric", "type": "nominal"},
        {"field": "Accuracy (%)", "type": "quantitative", "format": ".4f"},
        {
            "field": "Accuracy Drop (pp)",
            "type": "quantitative",
            "format": ".4f",
        },
        {
            "field": "Reference Accuracy (%)",
            "type": "quantitative",
            "format": ".4f",
        },
        {"field": "Target Layer Count", "type": "quantitative", "format": "d"},
        {"field": "MSE", "type": "quantitative", "format": ".3e"},
        {"field": "L1", "type": "quantitative", "format": ".3e"},
        {"field": "Run Date", "type": "nominal"},
        {"field": "Run ID", "type": "quantitative", "format": "d"},
    ]
    layers = [
        {
            "mark": {
                "type": "rule",
                "strokeDash": [7, 5],
                "strokeWidth": 1.5,
                "color": "#6b7280",
            },
            "encoding": {
                "y": {
                    "aggregate": "max",
                    "field": "Reference Line",
                    "type": "quantitative",
                }
            },
        }
    ]
    if view_mode == "Best per strategy":
        layers.append(
            {
                "mark": {"type": "line", "strokeWidth": 2.5},
                "encoding": {
                    **shared_encoding,
                    "color": {
                        "field": "Layer Type",
                        "type": "nominal",
                        "legend": {"title": "Layer type"},
                    },
                    "strokeDash": {
                        "field": "Strategy",
                        "type": "nominal",
                        "sort": LAYER_ABLATION_STRATEGY_ORDER,
                        "legend": {"title": "Strategy"},
                    },
                    "detail": {"field": "Series", "type": "nominal"},
                    "order": {"field": "Bits", "type": "quantitative"},
                },
            }
        )
    layers.append(
        {
            "mark": {"type": "point", "filled": True, "size": 105},
            "encoding": {
                **shared_encoding,
                "color": {
                    "field": "Layer Type",
                    "type": "nominal",
                    "legend": {"title": "Layer type"},
                },
                "shape": {
                    "field": "Strategy",
                    "type": "nominal",
                    "sort": LAYER_ABLATION_STRATEGY_ORDER,
                    "legend": {"title": "Strategy"},
                },
                "tooltip": tooltip,
            },
        }
    )
    chart_spec = {
        "config": {
            "view": {"stroke": "transparent"},
            "axis": {"labelFontSize": 11, "titleFontSize": 12},
            "legend": {"orient": "bottom", "columns": 4, "labelLimit": 220},
        },
        "layer": layers,
        "height": 480,
    }
    st.vega_lite_chart(chart_df, chart_spec, width="stretch")
    if view_mode == "Best per strategy":
        reference_note = (
            "The dashed rule is FP32 (or zero drop)."
            if not mixed_references or metric_label == "Top-1 accuracy drop"
            else "The FP32 rule is omitted because the logged references differ."
        )
        st.caption(
            "At each width, Uniform is the highest-accuracy fixed format; "
            "optimal points are the layer- and chunk-wise selections. "
            f"{reference_note}"
        )
    else:
        st.caption(
            "Every evaluated fixed format and optimal result is shown as a point. Hover for the exact format and run metadata."
        )

    st.markdown("#### Best result per strategy")
    summary_columns = [
        "Layer Type",
        "Bits",
        "Strategy",
        "Weight Format",
        "Optimization Metric",
        "Accuracy (%)",
        "Accuracy Drop (pp)",
        "Target Layer Count",
        "Evaluation Scope",
        "MSE",
        "L1",
        "Run ID",
    ]
    st.dataframe(best_df[summary_columns], width="stretch", hide_index=True)
    st.download_button(
        "Download filtered ablation results (CSV)",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_model}_layer_ablation_results.csv",
        mime="text/csv",
        key="download_layer_ablation_results",
    )
