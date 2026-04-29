with tab_graph:
    st.markdown("""
    <div class="dashboard-hero">
        <div class="dashboard-hero__eyebrow">Architecture · Quantization Map</div>
        <h1>Architecture Graph</h1>
        <p>Interactive model architecture viewer with quantization annotations. Generated live from the current code.</p>
    </div>
    """, unsafe_allow_html=True)

    if _graph_renderer_fn is None:
        st.info("No experiment runs found in the database. The graph viewer requires at least one run to initialize.")
    else:
        all_models_for_graph = sorted(cache_sim_df['model_name'].dropna().unique().tolist()) \
            if 'cache_sim_df' in dir() and not cache_sim_df.empty else []

        col_gm, col_gg, col_gr = st.columns([3, 1, 1])
        graph_model = col_gm.selectbox(
            "Model", all_models_for_graph or ["resnet18"],
            key="graph_tab_model_select",
        )
        generate_clicked = col_gg.button(
            "Load Graph", key="graph_tab_generate", type="primary", width='stretch'
        )
        regenerate_clicked = col_gr.button(
            "Regenerate", key="graph_tab_regenerate", width='stretch'
        )

        if generate_clicked or regenerate_clicked:
            force_regenerate = bool(regenerate_clicked)
            spinner_label = (
                f"Regenerating architecture graph for {graph_model}..."
                if force_regenerate or not use_cached_graphs
                else f"Loading architecture graph for {graph_model}..."
            )
            with st.spinner(spinner_label):
                try:
                    graph_json, graph_meta, graph_source = load_or_generate_model_graph_bundle(
                        graph_model,
                        use_cache=use_cached_graphs,
                        force_regenerate=force_regenerate,
                    )
                    st.session_state['_graph_tab_json'] = graph_json
                    st.session_state['_graph_tab_meta'] = graph_meta
                    st.session_state['_graph_tab_model'] = graph_model
                    st.session_state['_graph_tab_db'] = DB_PATH
                    if graph_source == "cache":
                        st.success("Loaded cached graph.")
                except Exception as exc:
                    st.error(f"Failed to load graph for `{graph_model}`: {exc}")

        cached_json  = st.session_state.get('_graph_tab_json')
        cached_meta  = st.session_state.get('_graph_tab_meta')
        cached_model = st.session_state.get('_graph_tab_model')
        cached_db = st.session_state.get('_graph_tab_db')

        if cached_json and cached_model == graph_model and cached_db == DB_PATH:
            _graph_renderer_fn(
                selected_model=graph_model,
                graph_json=cached_json,
                graph_meta=cached_meta,
                download_key="graph_tab_dl",
            )
        else:
            st.info("Select a model above and click **Load Graph**. Use **Regenerate** when the model code changed.")

st.sidebar.markdown("---")
st.sidebar.info("Managed via `src/database/handler.py`")
