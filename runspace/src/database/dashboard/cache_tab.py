import math
import os

@st.cache_data(ttl=30, show_spinner=False)
def _load_cache_sims(db_path):
    from runspace.src.database.handler import RunDatabase
    db = RunDatabase(db_path=db_path)
    return db.get_cache_simulations()


@st.cache_data(ttl=30, show_spinner=False)
def _load_bandwidth_aware_quant_results(project_root):
    results_roots = [
        (
            "bandwidth_aware_quant",
            os.path.join(project_root, "runspace/experiments/bandwidth_aware_quant/results"),
        ),
        (
            "baselines_vs_dynamic_runs",
            os.path.join(project_root, "runspace/experiments/baselines_vs_dynamic_runs/results/bandwidth_aware"),
        ),
    ]
    runs = []

    for source_label, results_root in results_roots:
        if not os.path.isdir(results_root):
            continue

        for dirpath, _, filenames in os.walk(results_root):
            if "bandwidth_aware_quant_results.json" not in filenames:
                continue
            json_path = os.path.join(dirpath, "bandwidth_aware_quant_results.json")
            rel_dir = os.path.relpath(dirpath, results_root)
            label = os.path.join(source_label, rel_dir)
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
            except Exception as exc:
                runs.append({
                    "label": label,
                    "path": json_path,
                    "error": str(exc),
                })
                continue

            runs.append({
                "label": label,
                "path": json_path,
                "dir": dirpath,
                "data": data,
                "model_name": data.get("model_name", rel_dir),
            })

    return sorted(runs, key=lambda r: r["label"])


@st.cache_data(ttl=30, show_spinner=False)
def _load_bandwidth_aware_descent_results(project_root, results_dir="results_descent"):
    """Load greedy-descent (--descent) runs, keyed by model name.

    Mirrors the regular results loader but scans a descent results directory.
    Each descent JSON carries its own `min_bits_sweeps` (the descent's winning
    acc-vs-speedup curve) plus a `descent` block, so it can be overlaid on the
    baseline chart.
    """
    root = os.path.join(project_root, "runspace/experiments/bandwidth_aware_quant", results_dir)
    runs = {}
    if not os.path.isdir(root):
        return runs
    for dirpath, _, filenames in os.walk(root):
        if "bandwidth_aware_quant_results.json" not in filenames:
            continue
        json_path = os.path.join(dirpath, "bandwidth_aware_quant_results.json")
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except Exception:
            continue
        model = data.get("model_name") or os.path.basename(dirpath)
        runs[model] = {"path": json_path, "data": data}
    return runs


def _bandwidth_aware_quant_rows(data, series="Baseline"):
    ref = data.get("ref_fp32", {}) or {}
    ref_acc = ref.get("accuracy")
    ref_cycles = {str(k): v for k, v in (ref.get("cycles_per_cache_size", {}) or {}).items()}
    ref_cycle_baseline = ref.get("baseline_cycles") or ref_cycles.get("0.0") or ref_cycles.get("0") or 1.0
    if ref_cycle_baseline <= 0:
        ref_cycle_baseline = 1.0
    rows = []

    for min_bits, cache_data in (data.get("min_bits_sweeps", {}) or {}).items():
        for cache_size, points in (cache_data or {}).items():
            for point in points or []:
                cycles = point.get("cycles")
                accuracy = point.get("accuracy")
                rows.append({
                    "min_bits": int(float(min_bits)),
                    "cache_size_M": float(cache_size),
                    "b": int(point.get("b")),
                    "accuracy": accuracy,
                    "acc_drop": (ref_acc - accuracy) if ref_acc is not None and accuracy is not None else None,
                    "cycles": cycles,
                    "ref_cycles": ref_cycle_baseline,
                    "speedup": (ref_cycle_baseline / cycles) if cycles else None,
                    "norm_speedup": (ref_cycle_baseline / cycles) if cycles else None,
                    "cache_label": f"Cache {float(cache_size):g}M",
                    "series": series,
                })

    return rows


def _render_bandwidth_aware_quant_chart(data, points_df, overlay=None):
    if points_df.empty:
        return

    def _mark_overlapping_series(df):
        """Mark duplicate cache-size series so the upper overlay can be dashed."""
        if df.empty:
            return df

        out = df.copy()
        out["line_style"] = "solid"
        out["overlap_note"] = ""
        if not {"cache_size_M", "b", "norm_speedup", "accuracy"}.issubset(out.columns):
            return out

        signatures = {}
        for cache_size, group in out.groupby("cache_size_M"):
            ordered = group.sort_values("b")
            signature = tuple(
                (
                    int(row["b"]),
                    round(float(row["norm_speedup"]), 9),
                    round(float(row["accuracy"]), 9),
                    round(float(row["cycles"]), 3) if row.get("cycles") is not None else None,
                )
                for _, row in ordered.iterrows()
            )
            signatures.setdefault(signature, []).append(cache_size)

        for cache_sizes in signatures.values():
            if len(cache_sizes) <= 1:
                continue
            for cache_size in sorted(cache_sizes)[1:]:
                mask = out["cache_size_M"] == cache_size
                out.loc[mask, "line_style"] = "dashed"
                out.loc[mask, "overlap_note"] = (
                    "same curve as smaller cache; dashed overlay"
                )
        return out

    try:
        import altair as alt
    except Exception:
        chart_df = points_df.rename(columns={"norm_speedup": "Normalized Speedup"})
        st.line_chart(chart_df, x="Normalized Speedup", y="accuracy", color="cache_label")
        return

    min_bits_values = sorted(points_df["min_bits"].dropna().unique().tolist())
    selected_min_bits = min_bits_values[0]
    if len(min_bits_values) > 1:
        selected_min_bits = st.selectbox(
            "Sweep",
            min_bits_values,
            format_func=lambda value: f"start >= {int(value)} bits",
            key="cache_bwaq_plot_sweep_select",
        )

    plot_df = points_df[points_df["min_bits"] == selected_min_bits].copy()
    if plot_df.empty:
        return

    def _fmt_cycles(value):
        if value is None or not math.isfinite(float(value)):
            return "N/A"
        value = float(value)
        if value >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"
        if value >= 1_000:
            return f"{value / 1_000:.1f}K"
        return f"{int(value)}"

    cycle_ranges = {}
    for cache_size, group in plot_df.groupby("cache_size_M"):
        cycles = group["cycles"].dropna()
        if cycles.empty:
            cycle_ranges[cache_size] = "N/A"
            continue
        cycle_ranges[cache_size] = (
            f"{_fmt_cycles(cycles.min())}-{_fmt_cycles(cycles.max())} cyc"
        )

    plot_df["point_type"] = "Quantized"
    plot_df = _mark_overlapping_series(plot_df)
    plot_df["bit_label"] = plot_df["b"].astype(int).astype(str)
    plot_df["legend_label"] = plot_df.apply(
        lambda row: f"Cache {float(row['cache_size_M']):g}M ({cycle_ranges.get(row['cache_size_M'], 'N/A')})",
        axis=1,
    )
    plot_df["cache_color"] = plot_df["legend_label"]

    ref = data.get("ref_fp32", {}) or {}
    ref_acc = ref.get("accuracy")
    ref_cycles = {str(k): v for k, v in (ref.get("cycles_per_cache_size", {}) or {}).items()}
    ref_cycle_baseline = ref.get("baseline_cycles") or ref_cycles.get("0.0") or ref_cycles.get("0") or 1.0
    if ref_cycle_baseline <= 0:
        ref_cycle_baseline = 1.0

    ref_rows = []
    if ref_acc is not None:
        cache_sizes = sorted(plot_df["cache_size_M"].dropna().unique().tolist())
        for cache_size in cache_sizes:
            cache_key = str(float(cache_size))
            fp32_cache_cycles = ref_cycles.get(cache_key) or ref_cycles.get(str(cache_size)) or ref_cycle_baseline
            fp32_cache_speedup = ref_cycle_baseline / fp32_cache_cycles if fp32_cache_cycles else None
            ref_rows.append({
                "min_bits": selected_min_bits,
                "cache_size_M": float(cache_size),
                "b": None,
                "accuracy": ref_acc,
                "cycles": fp32_cache_cycles,
                "norm_speedup": fp32_cache_speedup,
                "cache_label": f"Cache {float(cache_size):g}M",
                "cache_color": f"Cache {float(cache_size):g}M ({cycle_ranges.get(cache_size, 'N/A')})",
                "bit_label": "FP32",
                "legend_label": f"Ref FP32 {float(cache_size):g}M ({_fmt_cycles(fp32_cache_cycles)} cyc)",
                "point_type": "FP32 reference",
            })
    ref_df = pd.DataFrame(ref_rows)
    if not ref_df.empty:
        line_style_by_cache = plot_df.groupby("cache_size_M")["line_style"].first().to_dict()
        overlap_note_by_cache = plot_df.groupby("cache_size_M")["overlap_note"].first().to_dict()
        ref_df["line_style"] = ref_df["cache_size_M"].map(line_style_by_cache).fillna("solid")
        ref_df["overlap_note"] = ref_df["cache_size_M"].map(overlap_note_by_cache).fillna("")
    line_df = plot_df.copy()
    line_df["line_order"] = line_df["b"].astype(float)
    if not ref_df.empty:
        ref_line_df = ref_df.copy()
        ref_line_df["b"] = 32
        ref_line_df["bit_label"] = "32"
        ref_line_df["legend_label"] = ref_line_df["cache_color"]
        ref_line_df["line_order"] = 32.0
        line_df = pd.concat([line_df, ref_line_df], ignore_index=True)

    hover = alt.selection_point(
        fields=["cache_color"],
        on="pointerover",
        clear="pointerout",
        empty=True,
    )
    common_encoding = dict(
        x=alt.X(
            "norm_speedup:Q",
            title="Normalized Speedup vs FP32 0MB",
            scale=alt.Scale(zero=False),
        ),
        y=alt.Y(
            "accuracy:Q",
            title="Top-1 Accuracy (%)",
            scale=alt.Scale(zero=False),
        ),
        color=alt.Color("cache_color:N", title="Cache"),
        detail="legend_label:N",
        tooltip=[
            alt.Tooltip("cache_size_M:Q", title="Cache (M)", format=".1f"),
            alt.Tooltip("b:Q", title="Bits", format=".0f"),
            alt.Tooltip("accuracy:Q", title="Acc1 (%)", format=".3f"),
            alt.Tooltip("norm_speedup:Q", title="Norm Speedup", format=".3f"),
            alt.Tooltip("cycles:Q", title="Cycles", format=",.0f"),
            alt.Tooltip("overlap_note:N", title="Overlap"),
        ],
        opacity=alt.condition(hover, alt.value(1.0), alt.value(0.45)),
    )

    line_base = alt.Chart(line_df).encode(
        **common_encoding,
        order=alt.Order("line_order:Q", sort="descending"),
    )
    base = alt.Chart(plot_df).encode(**common_encoding)

    solid_lines = line_base.transform_filter(
        alt.datum.line_style != "dashed"
    ).mark_line(point=False, strokeWidth=2.5).add_params(hover)
    dashed_lines = line_base.transform_filter(
        alt.datum.line_style == "dashed"
    ).mark_line(point=False, strokeWidth=3.0, strokeDash=[7, 4]).add_params(hover)
    points = base.mark_circle(size=180)
    label_halo = base.mark_text(
        align="center",
        baseline="middle",
        color="white",
        fontWeight="bold",
        fontSize=12,
        stroke="black",
        strokeWidth=4,
    ).encode(text="bit_label:N")
    labels = base.mark_text(
        align="center",
        baseline="middle",
        color="white",
        fontWeight="bold",
        fontSize=12,
    ).encode(text="bit_label:N")

    layers = [solid_lines, dashed_lines, points, label_halo, labels]
    if not ref_df.empty:
        ref_points = alt.Chart(ref_df).mark_point(
            shape="diamond",
            filled=True,
            size=170,
        ).encode(
            x="norm_speedup:Q",
            y="accuracy:Q",
            color=alt.Color("cache_color:N", title="Cache"),
            shape=alt.Shape("legend_label:N", title="Run"),
            tooltip=[
                alt.Tooltip("cache_size_M:Q", title="Cache (M)", format=".1f"),
                alt.Tooltip("point_type:N", title="Type"),
                alt.Tooltip("accuracy:Q", title="Acc1 (%)", format=".3f"),
                alt.Tooltip("norm_speedup:Q", title="Norm Speedup", format=".3f"),
                alt.Tooltip("cycles:Q", title="Cycles", format=",.0f"),
                alt.Tooltip("overlap_note:N", title="Overlap"),
            ],
        )
        ref_labels = alt.Chart(ref_df).mark_text(
            align="left",
            dx=9,
            dy=-7,
            fontSize=10,
            color="#202124",
        ).encode(
            x="norm_speedup:Q",
            y="accuracy:Q",
            text="bit_label:N",
        )
        layers.extend([ref_points, ref_labels])

    # Optional greedy-descent overlays: same axes, distinct markers + dashed
    # lines, and their own colors so they read as separate curves "on top".
    if overlay is not None:
        overlays = overlay if isinstance(overlay, list) else [overlay]
        for overlay_spec in overlays:
            if isinstance(overlay_spec, dict):
                overlay_rows_df = overlay_spec.get("rows")
                point_type = overlay_spec.get("point_type", "Descent (8→3)")
                legend_prefix = overlay_spec.get("legend_prefix", "Descent")
                marker_shape = overlay_spec.get("shape", "square")
                stroke_dash = overlay_spec.get("stroke_dash", [6, 3])
            else:
                _overlay_data, overlay_rows_df = overlay_spec
                point_type = "Descent (8→3)"
                legend_prefix = "Descent"
                marker_shape = "square"
                stroke_dash = [6, 3]

            if overlay_rows_df is None or overlay_rows_df.empty:
                continue

            o = overlay_rows_df[overlay_rows_df["min_bits"] == selected_min_bits].copy()
            if not o.empty:
                o_ranges = {}
                for cache_size, group in o.groupby("cache_size_M"):
                    o_cycles = group["cycles"].dropna()
                    o_ranges[cache_size] = (
                        f"{_fmt_cycles(o_cycles.min())}-{_fmt_cycles(o_cycles.max())} cyc"
                        if not o_cycles.empty else "N/A"
                    )
                o["point_type"] = point_type
                o = _mark_overlapping_series(o)
                o["bit_label"] = o["b"].astype(int).astype(str)
                o["legend_label"] = o.apply(
                    lambda row: f"{legend_prefix} {float(row['cache_size_M']):g}M ({o_ranges.get(row['cache_size_M'], 'N/A')})",
                    axis=1,
                )
                o["cache_color"] = o["legend_label"]
                o_line = o.copy()
                o_line["line_order"] = o_line["b"].astype(float)

                o_base = alt.Chart(o).encode(**common_encoding)
                o_line_base = alt.Chart(o_line).encode(
                    **common_encoding,
                    order=alt.Order("line_order:Q", sort="descending"),
                )
                o_lines = o_line_base.mark_line(
                    point=False, strokeWidth=2.5, strokeDash=stroke_dash
                ).add_params(hover)
                o_points = o_base.mark_point(shape=marker_shape, filled=True, size=150)
                o_halo = o_base.mark_text(
                    align="center", baseline="middle", color="white",
                    fontWeight="bold", fontSize=11, stroke="black", strokeWidth=4,
                ).encode(text="bit_label:N")
                o_labels = o_base.mark_text(
                    align="center", baseline="middle", color="white",
                    fontWeight="bold", fontSize=11,
                ).encode(text="bit_label:N")
                layers.extend([o_lines, o_points, o_halo, o_labels])

    chart = alt.layer(*layers).properties(height=420).interactive()
    st.altair_chart(chart, use_container_width=True)


def _render_descent_policy_table(descent_block, label="Descent"):
    """Render the descent's chosen weight policy per bit-width, one column per cache.

    `descent_block` is data["descent"]: {cache_str: {"policy_by_bits": {bits: policy}, ...}}.
    Rows are bit-widths 8→3; cells are the winning fixed format or "mse" (per-chunk MSE).
    """
    if not descent_block:
        return

    cache_keys = sorted(descent_block.keys(), key=lambda c: float(c))
    all_bits = set()
    for cache in cache_keys:
        all_bits.update(int(b) for b in (descent_block[cache].get("policy_by_bits", {}) or {}))
    if not all_bits:
        return

    table = {}
    for cache in cache_keys:
        pol = descent_block[cache].get("policy_by_bits", {}) or {}
        col = f"Cache {float(cache):g}M"
        table[col] = {f"{b}-bit": pol.get(str(b), pol.get(b, "—")) for b in sorted(all_bits, reverse=True)}

    policy_df = pd.DataFrame(table)
    st.caption(f"{label} chosen weight policy per bit-width (fixed format or `mse` = per-chunk MSE)")
    st.dataframe(policy_df, width='stretch')


def _render_bandwidth_aware_quant_results(selected_model=None):
    runs = _load_bandwidth_aware_quant_results(PROJECT_ROOT)
    if not runs:
        st.info(
            "No bandwidth-aware quant results found. "
            "Run `runspace/experiments/bandwidth_aware_quant/bandwidth_aware_quant.py` first."
        )
        return

    labels = [run["label"] for run in runs]
    default_idx = 0
    if selected_model:
        for idx, run in enumerate(runs):
            if run.get("model_name") == selected_model or run["label"].endswith(selected_model):
                default_idx = idx
                break

    selected_label = st.selectbox(
        "Bandwidth-aware result",
        labels,
        index=default_idx,
        key="cache_bwaq_result_select",
    )
    run = runs[labels.index(selected_label)]

    if run.get("error"):
        st.error(f"Could not parse {run['path']}: {run['error']}")
        return

    data = run["data"]
    ref = data.get("ref_fp32", {}) or {}
    rows = _bandwidth_aware_quant_rows(data)
    points_df = pd.DataFrame(rows)

    st.caption(run["path"])
    m1, m2, m3 = st.columns(3)
    m1.metric("Model", data.get("model_name", "N/A"))
    m2.metric("FP32 Acc1", f"{float(ref.get('accuracy', 0.0)):.3f}%")
    m3.metric("Sweep Points", len(points_df))

    if not points_df.empty:
        sort_cols = [c for c in ["min_bits", "cache_size_M", "b"] if c in points_df.columns]
        points_df = points_df.sort_values(sort_cols)
        st.markdown("##### Accuracy vs Speedup")

        overlays = []
        descent_runs = _load_bandwidth_aware_descent_results(PROJECT_ROOT)
        descent_run = descent_runs.get(data.get("model_name"))
        if descent_run is not None and not bool(data.get("used_descent")):
            if st.checkbox(
                "Overlay greedy-descent (8→3) results",
                value=False,
                key="cache_bwaq_overlay_descent",
                help=f"Overlay the matching descent run on top: {descent_run['path']}",
            ):
                overlays.append({
                    "data": descent_run["data"],
                    "rows": pd.DataFrame(_bandwidth_aware_quant_rows(descent_run["data"], series="Descent")),
                    "point_type": "Descent (8→3)",
                    "legend_prefix": "Descent",
                    "shape": "square",
                    "stroke_dash": [6, 3],
                })
                _render_descent_policy_table(descent_run["data"].get("descent", {}), label="Descent")

        e1e2_descent_runs = _load_bandwidth_aware_descent_results(
            PROJECT_ROOT,
            results_dir="results_descent_activation_e1e2",
        )
        e1e2_descent_run = e1e2_descent_runs.get(data.get("model_name"))
        if e1e2_descent_run is not None and not bool(data.get("used_descent")):
            if st.checkbox(
                "Overlay e1/e2 activation descent results",
                value=False,
                key="cache_bwaq_overlay_descent_activation_e1e2",
                help=f"Overlay the matching e1/e2 activation descent run on top: {e1e2_descent_run['path']}",
            ):
                overlays.append({
                    "data": e1e2_descent_run["data"],
                    "rows": pd.DataFrame(_bandwidth_aware_quant_rows(e1e2_descent_run["data"], series="Descent e1/e2 activations")),
                    "point_type": "Descent (e1/e2 activations)",
                    "legend_prefix": "Descent e1/e2",
                    "shape": "triangle-up",
                    "stroke_dash": [2, 3],
                })
                _render_descent_policy_table(
                    e1e2_descent_run["data"].get("descent", {}),
                    label="Descent e1/e2 activations",
                )

        _render_bandwidth_aware_quant_chart(data, points_df, overlay=overlays or None)

        hidden_cols = {"min_bits", "norm_speedup", "cache_label", "series"}
        visible_cols = [c for c in points_df.columns if c not in hidden_cols]
        st.dataframe(
            points_df[visible_cols],
            width='stretch',
            hide_index=True,
            column_config={
                "cache_size_M": st.column_config.NumberColumn("Cache (M)", format="%.1f", width="small"),
                "b":            st.column_config.NumberColumn("B", format="%d", width="small"),
                "accuracy":     st.column_config.NumberColumn("Acc1 (%)", format="%.3f", width="small"),
                "acc_drop":     st.column_config.NumberColumn("Drop (%)", format="%.3f", width="small"),
                "cycles":       st.column_config.NumberColumn("Cycles", format="%d", width="medium"),
                "ref_cycles":   st.column_config.NumberColumn("Ref Cycles", format="%d", width="medium"),
                "speedup":      st.column_config.NumberColumn("Speedup", format="%.3f×", width="small"),
            },
        )
    else:
        st.info("This result file has no sweep points.")

def _render_cache_run_details(cache_sim_df):
    if cache_sim_df.empty:
        st.info(
            "No cache simulations in the database yet. "
            "Run `simulate_cache.py` to populate this tab."
        )
        return

    available_models = sorted(cache_sim_df['model_name'].dropna().unique())
    col_sel, col_refresh = st.columns([4, 1])
    selected_model = col_sel.selectbox(
        "Model", available_models, key="cache_sim_model_select"
    )
    if col_refresh.button("Refresh", key="cache_sim_refresh", width='stretch'):
        _load_cache_sims.clear()
        rerun_current_fragment()

    model_rows = cache_sim_df[cache_sim_df['model_name'] == selected_model].sort_values('id', ascending=False)
    latest = model_rows.iloc[0]

    if len(model_rows) > 1:
        run_options = [
            f"Run {row['id']}  -  {row['timestamp'] or ''}  |  cache={row['cache_size_M']}M  banks={row['num_banks']}"
            for _, row in model_rows.iterrows()
        ]
        chosen_idx = st.selectbox(
            "Simulation run",
            range(len(run_options)),
            format_func=lambda i: run_options[i],
            key="cache_sim_run_select",
        )
        latest = model_rows.iloc[chosen_idx]

    total = int(latest.get('total_layers') or 0)
    off_chip = int(latest.get('off_chip_count') or 0)
    flagged = int(latest.get('flagged_count') or 0)
    on_chip = total - off_chip - flagged
    num_banks = int(latest.get('num_banks') or 16)
    bank_size = int(latest.get('bank_size') or 1)
    cache_elements = int(num_banks * bank_size)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Layers", total)
    m2.metric("On Chip", on_chip)
    m3.metric("Off Chip (QUANT)", off_chip, delta=f"-{off_chip}" if off_chip else None, delta_color="inverse")
    m4.metric("Flagged", flagged, delta=f"-{flagged}" if flagged else None, delta_color="inverse")
    m5.metric("Cache Size", f"{latest.get('cache_size_M', '?')}M  /{num_banks} banks")

    st.markdown(
        f"<p class='dashboard-filter-note'>metadata_bits={int(latest.get('metadata_bits') or 0)} &nbsp;·&nbsp; "
        f"bandwidth={latest.get('bandwidth', 1.0):.1f} B/cyc &nbsp;·&nbsp; "
        f"bank_size={bank_size:,} elem &nbsp;·&nbsp; "
        f"timestamp={latest.get('timestamp', '—')}</p>",
        unsafe_allow_html=True,
    )

    layers_raw = latest.get('layers_json') or '[]'
    layers = json.loads(layers_raw) if isinstance(layers_raw, str) else layers_raw

    if layers:
        st.markdown("#### Memory Bank View")
        st.caption(
            "Step through layers to see which banks are occupied and by what. "
            "Use the slider or Prev/Next buttons."
        )
        rules_raw = latest.get('rules_json') or '[]'
        rules_list = json.loads(rules_raw) if isinstance(rules_raw, str) else (rules_raw or [])
        rule_meta = {r['name']: r for r in rules_list} if rules_list else None
        states = _build_bank_states(layers, num_banks, bank_size, rule_meta=rule_meta)
        import streamlit.components.v1 as _comp
        _comp.html(
            _render_bank_viz_html(states, num_banks, bank_size, cache_elements),
            height=420,
            scrolling=False,
        )

        st.markdown("---")
        st.markdown("#### Layer Details")

        formatted_layers = []
        for layer in layers:
            layer_copy = dict(layer)
            if 'residual_connections' not in layer_copy and layer.get('type') == 'QuantAdd':
                out_elems = float(layer.get('output_elems') or 0)
                base_cycles = math.ceil(out_elems / 128) if out_elems > 0 else 0
                compute_cycles = float(layer.get('compute_cycles') or 0)
                layer_copy['residual_connections'] = (
                    int(round(compute_cycles / base_cycles))
                    if base_cycles > 0 and compute_cycles > 0 else 1
                )
            collapsed = layer.get('collapsed_layers', [])
            if collapsed:
                collapsed_names = ", ".join(c['name'] for c in collapsed)
                collapsed_types = ", ".join(c['type'] for c in collapsed)
                layer_copy['name'] = f"{layer['name']} (+ {collapsed_names})"
                layer_copy['type'] = f"{layer['type']} (+ {collapsed_types})"
            formatted_layers.append(layer_copy)

        layers_df = pd.DataFrame(formatted_layers)
        display_cols = [c for c in [
            'name', 'type', 'stay_on_chip', 'rule', 'reason',
            'residual_connections',
            'input_elems', 'weight_elems', 'output_elems',
            'output_banked', 'next_xin_banked', 'next_layer_name',
            'input_bits', 'weight_bits', 'output_bits',
            'residual_input_elems', 'residual_input_bits',
            'residual_output_elems', 'residual_output_bits',
            'input_bw_limited', 'weight_bw_limited', 'output_bw_limited',
            'residual_input_bw_limited', 'residual_output_bw_limited',
            'compute_cycles', 'total_cycles',
        ] if c in layers_df.columns]

        layers_viz = layers_df[display_cols].copy()
        for col in [
            'input_elems', 'weight_elems', 'output_elems',
            'residual_input_elems', 'residual_output_elems',
            'output_banked', 'next_xin_banked',
        ]:
            if col in layers_viz.columns:
                layers_viz[col] = layers_viz[col].astype(float) / 1000.0

        for col in ['compute_cycles', 'total_cycles']:
            if col in layers_viz.columns:
                layers_viz[col] = layers_viz[col].fillna(0).astype(int)

        st.dataframe(
            layers_viz,
            width='stretch',
            height=500,
            column_config={
                'name':            st.column_config.TextColumn("Layer",        width="large"),
                'type':            st.column_config.TextColumn("Type",         width="small"),
                'stay_on_chip':    st.column_config.CheckboxColumn("On Chip",  width="small"),
                'rule':            st.column_config.TextColumn("Rule",         width="medium"),
                'reason':          st.column_config.TextColumn("Reason",       width="large"),
                'residual_connections': st.column_config.NumberColumn("Res Adds", format="%d", width="small"),
                'input_elems':     st.column_config.NumberColumn("Input (K)",    format="%.1f K", width="small"),
                'weight_elems':    st.column_config.NumberColumn("Weights (K)",  format="%.1f K", width="small"),
                'output_elems':    st.column_config.NumberColumn("Output (K)",   format="%.1f K", width="small"),
                'output_banked':   st.column_config.NumberColumn("Banked (K)",   format="%.1f K", width="small"),
                'next_xin_banked': st.column_config.NumberColumn("Next Xin (K)", format="%.1f K", width="small"),
                'next_layer_name': st.column_config.TextColumn("Next Layer",   width="large"),
                'input_bits':      st.column_config.NumberColumn("in Bits",      format="%d", width="small"),
                'weight_bits':     st.column_config.NumberColumn("W Bits",       format="%d", width="small"),
                'output_bits':     st.column_config.NumberColumn("out Bits",     format="%d", width="small"),
                'residual_input_elems':  st.column_config.NumberColumn("res_in (K)",    format="%.1f K", width="small"),
                'residual_input_bits':   st.column_config.NumberColumn("res_in Bits",   format="%d", width="small"),
                'residual_output_elems': st.column_config.NumberColumn("output_res (K)", format="%.1f K", width="small"),
                'residual_output_bits':  st.column_config.NumberColumn("res_out Bits",   format="%d", width="small"),
                'input_bw_limited':   st.column_config.CheckboxColumn("xin BW",  width="small"),
                'weight_bw_limited':  st.column_config.CheckboxColumn("W BW",    width="small"),
                'output_bw_limited':  st.column_config.CheckboxColumn("xout BW", width="small"),
                'residual_input_bw_limited':  st.column_config.CheckboxColumn("res_in BW", width="small"),
                'residual_output_bw_limited': st.column_config.CheckboxColumn("xres BW", width="small"),
                'compute_cycles':  st.column_config.NumberColumn("Comp Cyc",     format="%d", width="small"),
                'total_cycles':    st.column_config.NumberColumn("Total Cyc",    format="%d", width="small"),
            },
        )

        off_chip_raw = latest.get('off_chip_layers_json') or '[]'
        off_chip_list = json.loads(off_chip_raw) if isinstance(off_chip_raw, str) else off_chip_raw
        if off_chip_list:
            with st.expander(f"Off-chip layers ({len(off_chip_list)}) - copy for runner config"):
                st.code(json.dumps(off_chip_list, indent=2), language="json")

    st.markdown("---")
    st.markdown("#### Rule Reference")
    st.caption(
        "Parsed from the simulation record in the DB - always reflects the rules that were "
        "actually used. Re-run `simulate_cache.py` after changing rule definitions to update."
    )

    rules_raw = latest.get('rules_json')
    rules_list = json.loads(rules_raw) if isinstance(rules_raw, str) and rules_raw else []

    if not rules_list:
        st.info(
            "No rule metadata in this simulation record. "
            "Re-run `simulate_cache.py` to populate the rules table."
        )
        return

    rules_df = pd.DataFrame(rules_list).rename(columns={
        'name':           'Rule',
        'on_chip':        'xout On-Chip',
        'xin_from_cache': 'xin Source',
        'applies_to':     'Applies To',
        'stay_condition': 'Stay Condition',
        'permanents':     'Permanents',
        'notes':          'Notes',
    })
    if 'xout On-Chip' in rules_df.columns:
        rules_df['xout On-Chip'] = rules_df['xout On-Chip'].map(
            {True: 'Yes', False: 'No', 'True': 'Yes', 'False': 'No'}
        ).fillna(rules_df['xout On-Chip'].astype(str))
    if 'xin Source' in rules_df.columns:
        rules_df['xin Source'] = rules_df['xin Source'].map(
            {True: 'Cache', False: 'External', 'True': 'Cache', 'False': 'External'}
        ).fillna(rules_df['xin Source'].astype(str))

    display_cols = [c for c in [
        'Rule', 'xout On-Chip', 'xin Source',
        'Applies To', 'Stay Condition', 'Permanents', 'Notes',
    ] if c in rules_df.columns]

    st.dataframe(
        rules_df[display_cols],
        width='stretch',
        hide_index=True,
        column_config={
            'Rule':           st.column_config.TextColumn("Rule",            width="medium"),
            'xout On-Chip':   st.column_config.TextColumn("xout On-Chip",   width="small"),
            'xin Source':     st.column_config.TextColumn("xin Source",     width="small"),
            'Applies To':     st.column_config.TextColumn("Applies To",     width="medium"),
            'Stay Condition': st.column_config.TextColumn("Stay Condition", width="large"),
            'Permanents':     st.column_config.TextColumn("Permanents",     width="medium"),
            'Notes':          st.column_config.TextColumn("Notes",          width="large"),
        },
    )


with tab_cache:
    @st.fragment
    def _render_cache_tab():
        cache_sim_df = _load_cache_sims(DB_PATH)

        st.markdown("""
        <div class="dashboard-hero">
            <div class="dashboard-hero__eyebrow">ASIC · On-Chip Memory</div>
            <h1>Cache Simulation</h1>
            <p>Layer-by-layer cache placement decisions — which outputs stay on chip and which must be quantized for external memory transfer.</p>
        </div>
        """, unsafe_allow_html=True)

        cache_runs_tab, bwaq_tab = st.tabs(["Cache Runs", "Bandwidth-Aware Quant"])
        with cache_runs_tab:
            _render_cache_run_details(cache_sim_df)
        with bwaq_tab:
            _render_bandwidth_aware_quant_results()

    _render_cache_tab()

# ── Architecture Graph Tab ───────────────────────────────────────────────────
