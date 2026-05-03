@st.cache_data(ttl=30, show_spinner=False)
def _load_cache_sims(db_path):
    from runspace.src.database.handler import RunDatabase
    db = RunDatabase(db_path=db_path)
    return db.get_cache_simulations()

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

        if cache_sim_df.empty:
            st.info(
                "No cache simulations in the database yet. "
                "Run `simulate_cache.py` to populate this tab."
            )
            return

        # ── Model selector ────────────────────────────────────────────────────
        available_models = sorted(cache_sim_df['model_name'].dropna().unique())
        col_sel, col_refresh = st.columns([4, 1])
        selected_model = col_sel.selectbox(
            "Model", available_models, key="cache_sim_model_select"
        )
        if col_refresh.button("🔄 Refresh", key="cache_sim_refresh", width='stretch'):
            _load_cache_sims.clear()
            rerun_current_fragment()

        # Latest simulation row for the selected model
        model_rows = cache_sim_df[cache_sim_df['model_name'] == selected_model].sort_values('id', ascending=False)
        latest = model_rows.iloc[0]

        # Show all runs for this model (for historical comparison)
        if len(model_rows) > 1:
            run_options = [
                f"Run {row['id']}  —  {row['timestamp'] or ''}  |  cache={row['cache_size_M']}M  banks={row['num_banks']}"
                for _, row in model_rows.iterrows()
            ]
            chosen_idx = st.selectbox("Simulation run", range(len(run_options)),
                                      format_func=lambda i: run_options[i],
                                      key="cache_sim_run_select")
            latest = model_rows.iloc[chosen_idx]

        # ── Summary metrics ───────────────────────────────────────────────────
        total    = int(latest.get('total_layers') or 0)
        off_chip = int(latest.get('off_chip_count') or 0)
        flagged  = int(latest.get('flagged_count') or 0)
        on_chip  = total - off_chip - flagged
        num_banks = int(latest.get('num_banks') or 16)
        bank_size = int(latest.get('bank_size') or 1)
        cache_elements = int(num_banks * bank_size)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Layers",   total)
        m2.metric("On Chip",        on_chip)
        m3.metric("Off Chip (QUANT)", off_chip,
                  delta=f"-{off_chip}" if off_chip else None, delta_color="inverse")
        m4.metric("Flagged",        flagged,
                  delta=f"-{flagged}" if flagged else None, delta_color="inverse")
        m5.metric("Cache Size",     f"{latest.get('cache_size_M', '?')}M  ×{num_banks} banks")

        st.markdown(
            f"<p class='dashboard-filter-note'>metadata_bits={int(latest.get('metadata_bits') or 0)} &nbsp;·&nbsp; "
            f"bank_size={bank_size:,} elem &nbsp;·&nbsp; "
            f"timestamp={latest.get('timestamp', '—')}</p>",
            unsafe_allow_html=True,
        )

        # ── Parse layers (keep raw ints for viz, format copy for table) ───────
        layers_raw = latest.get('layers_json') or '[]'
        layers = json.loads(layers_raw) if isinstance(layers_raw, str) else layers_raw

        if layers:
            # ── Memory Bank Visualizer ────────────────────────────────────────
            st.markdown("#### Memory Bank View")
            st.caption(
                "Step through layers to see which banks are occupied and by what. "
                "Use the slider or Prev/Next buttons."
            )
            rules_raw  = latest.get('rules_json') or '[]'
            rules_list = json.loads(rules_raw) if isinstance(rules_raw, str) else (rules_raw or [])
            rule_meta  = {r['name']: r for r in rules_list} if rules_list else None
            states = _build_bank_states(layers, num_banks, bank_size, rule_meta=rule_meta)
            import streamlit.components.v1 as _comp
            _comp.html(
                _render_bank_viz_html(states, num_banks, bank_size, cache_elements),
                height=420,
                scrolling=False,
            )

            st.markdown("---")

            # ── Layer table ───────────────────────────────────────────────────
            st.markdown("#### Layer Details")
            layers_df = pd.DataFrame(layers)
            fmt_cols = ['input_elems', 'weight_elems', 'output_elems',
                        'output_banked', 'next_xin_banked', 'perm_elems']
            layers_fmt = layers_df.copy()
            for col in fmt_cols:
                if col in layers_fmt.columns:
                    layers_fmt[col] = layers_fmt[col].apply(
                        lambda n: _fmt_e(n) if pd.notna(n) else ""
                    )

            display_cols = [c for c in [
                'name', 'type', 'stay_on_chip', 'rule', 'reason',
                'input_elems', 'weight_elems', 'output_elems',
                'output_banked', 'next_xin_banked', 'next_layer_name',
            ] if c in layers_fmt.columns]

            st.dataframe(
                layers_fmt[display_cols],
                width='stretch',
                height=500,
                column_config={
                    'name':            st.column_config.TextColumn("Layer",        width="large"),
                    'type':            st.column_config.TextColumn("Type",         width="small"),
                    'stay_on_chip':    st.column_config.CheckboxColumn("On Chip",  width="small"),
                    'rule':            st.column_config.TextColumn("Rule",         width="medium"),
                    'reason':          st.column_config.TextColumn("Reason",       width="large"),
                    'input_elems':     st.column_config.TextColumn("Input",        width="small"),
                    'weight_elems':    st.column_config.TextColumn("Weights",      width="small"),
                    'output_elems':    st.column_config.TextColumn("Output",       width="small"),
                    'output_banked':   st.column_config.TextColumn("Banked",       width="small"),
                    'next_xin_banked': st.column_config.TextColumn("Next Xin",    width="small"),
                    'next_layer_name': st.column_config.TextColumn("Next Layer",   width="large"),
                },
            )

            # Off-chip layers list
            off_chip_raw = latest.get('off_chip_layers_json') or '[]'
            off_chip_list = json.loads(off_chip_raw) if isinstance(off_chip_raw, str) else off_chip_raw
            if off_chip_list:
                with st.expander(f"Off-chip layers ({len(off_chip_list)}) — copy for runner config"):
                    st.code(json.dumps(off_chip_list, indent=2), language="json")

        st.markdown("---")

        # ── Rules Reference — fetched from DB, reflects the rules used at run time ─
        st.markdown("#### Rule Reference")
        st.caption(
            "Parsed from the simulation record in the DB — always reflects the rules that were "
            "actually used. Re-run `simulate_cache.py` after changing rule definitions to update."
        )

        rules_raw  = latest.get('rules_json')
        rules_list = json.loads(rules_raw) if isinstance(rules_raw, str) and rules_raw else []

        if not rules_list:
            st.info(
                "No rule metadata in this simulation record. "
                "Re-run `simulate_cache.py` to populate the rules table."
            )
        else:
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
                    {True: '✓ Yes', False: '✗ No', 'True': '✓ Yes', 'False': '✗ No'}
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

    _render_cache_tab()

# ── Architecture Graph Tab ───────────────────────────────────────────────────
