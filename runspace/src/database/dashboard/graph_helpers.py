def generate_live_model_graph_bundle(model_name, graph_depth=12):
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is not available in this environment.")
    if not GRAPH_VIZ_AVAILABLE:
        raise RuntimeError("Graph visualization dependencies are not available.")

    adapter = GenericAdapter(
        model_name=model_name,
        quantized_ops=["all"],
        input_quantization=False,
        enable_fx_quantization=False,
    )
    model = adapter.model
    model.eval()

    graph_json = generate_hierarchical_json(
        model,
        input_size=(1, 3, 224, 224),
        model_name=model_name,
        depth=graph_depth,
    )

    parsed_json = json.loads(graph_json)
    num_nodes = sum(1 for e in parsed_json if e.get('data', {}).get('type') in ('node', 'compound'))
    num_quantized = sum(1 for e in parsed_json if '#a7f3d0' in str(e.get('data', {}).get('color', '')))

    return graph_json, {
        'model_name': model_name,
        'graph_size_original': len(graph_json.encode('utf-8')),
        'num_nodes': num_nodes,
        'num_quantized_layers': num_quantized,
        'graph_depth': graph_depth,
        'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'status': 'LIVE',
    }


@st.cache_data(show_spinner=False)
def get_cached_model_graph_bundle(db_path, model_name):
    db = RunDatabase(db_path=db_path)
    graph_json, graph_meta = db.get_model_graph_json(model_name)
    if graph_json and graph_meta is not None:
        graph_meta = dict(graph_meta)
        graph_meta['source'] = 'cache'
    return graph_json, graph_meta


def load_or_generate_model_graph_bundle(model_name, use_cache=True, force_regenerate=False):
    if use_cache and not force_regenerate:
        graph_json, graph_meta = get_cached_model_graph_bundle(DB_PATH, model_name)
        if graph_json:
            return graph_json, graph_meta, "cache"

    graph_json, graph_meta = generate_live_model_graph_bundle(model_name)
    graph_meta = dict(graph_meta or {})
    graph_meta['source'] = 'live'
    if use_cache:
        # Avoid keeping an earlier cache miss around after this model is stored.
        get_cached_model_graph_bundle.clear()
        db = RunDatabase(db_path=DB_PATH)
        db.store_model_graph(model_name, graph_json, graph_meta)
    return graph_json, graph_meta, "live"


def _render_win_rates(raw_map_json, title_prefix, empty_info):
    summary_df, layer_df, layer_chunk_df, meta = _compute_weight_win_rate_views(raw_map_json)
    if summary_df is None:
        st.info(empty_info)
        return

    st.markdown(f"#### 🏆 {title_prefix} Chunk / Layer Win Rates")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Layers", f"{meta['layers']}")
    m2.metric("Chunks", f"{meta['chunks']}")
    m3.metric("Top Layer Winner", meta["top_layer_format"])
    m4.metric("Top Chunk Winner", meta["top_chunk_format"])

    # Stacked bar chart similar to plot_chunk_win_rate in experiments utils.
    if layer_chunk_df is not None and not layer_chunk_df.empty:
        st.markdown("**Chunk format selection per layer**")
        chart_df = layer_chunk_df.copy()
        format_totals = chart_df.groupby("Format")["Chunk Wins"].sum().to_dict()
        active_formats = [
            f for f in _sort_quant_formats(format_totals.keys())
            if int(format_totals.get(f, 0)) > 0
        ]
        chart_df = chart_df[chart_df["Format"].isin(active_formats)]
        layer_order = layer_df.sort_values(by=["Layer Index", "Layer"])["Layer"].tolist()
        st.vega_lite_chart(
            chart_df,
            {
                "mark": {"type": "bar", "tooltip": True},
                "encoding": {
                    "x": {
                        "field": "Layer",
                        "type": "nominal",
                        "axis": {"labelAngle": -90, "title": ""},
                        "sort": layer_order,
                    },
                    "y": {
                        "field": "Chunk Wins",
                        "type": "quantitative",
                        "stack": "zero",
                        "title": "Number of Chunks",
                    },
                    "color": {
                        "field": "Format",
                        "type": "nominal",
                        "sort": active_formats,
                        "legend": {"orient": "right"},
                    },
                },
                "height": 360,
            },
            width='stretch',
        )

    st.dataframe(
        summary_df,
        width='stretch',
        column_config={
            "Format": st.column_config.TextColumn("Format"),
            "Layer Wins": st.column_config.NumberColumn("Layer Wins", format="%d"),
            "Layer Win Rate (%)": st.column_config.NumberColumn("Layer Win Rate (%)", format="%.2f"),
            "Chunk Wins": st.column_config.NumberColumn("Chunk Wins", format="%d"),
            "Chunk Win Rate (%)": st.column_config.NumberColumn("Chunk Win Rate (%)", format="%.2f"),
        }
    )

    with st.expander("Per-layer dominant winners", expanded=False):
        st.dataframe(
            layer_df[["Layer", "Type", "Dominant Format", "Chunks"]],
            width='stretch',
            column_config={
                "Layer": st.column_config.TextColumn("Layer", width="large"),
                "Type": st.column_config.TextColumn("Type"),
                "Dominant Format": st.column_config.TextColumn("Dominant Format"),
                "Chunks": st.column_config.NumberColumn("Chunks", format="%d"),
            }
        )


def _render_weight_win_rates(raw_quant_map_json):
    _render_win_rates(
        raw_map_json=raw_quant_map_json,
        title_prefix="Weight",
        empty_info="No weight quantization map stored for this run.",
    )


def _render_input_win_rates(raw_input_map_json):
    _render_win_rates(
        raw_map_json=raw_input_map_json,
        title_prefix="Activation/Input",
        empty_info="No activation/input quantization map stored for this run.",
    )


def _is_input_experiment(experiment_type):
    text = str(experiment_type or "").strip().lower()
    return text.startswith("input_quant")


def _resolve_maps_for_display(run_row):
    """
    Return (weight_map_json, input_map_json, is_input_experiment).

    For input-only experiments we intentionally suppress weight map rendering.
    Legacy rows may have input maps mirrored into quant_map_json, so we fall back
    to quant_map_json only when input_map_json is absent.
    """
    raw_weight_map = run_row.get('quant_map_json')
    raw_input_map = run_row.get('input_map_json')
    is_input_exp = _is_input_experiment(run_row.get('experiment_type'))

    if is_input_exp:
        if _safe_json_load(raw_input_map) is None and _safe_json_load(raw_weight_map) is not None:
            raw_input_map = raw_weight_map
        raw_weight_map = None

    return raw_weight_map, raw_input_map, is_input_exp


def render_dashboard_intro():
    st.markdown(
        """
        <div class="dashboard-hero">
            <h1>QBench Experiment Dashboard</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


