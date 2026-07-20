with tab_bwaware_best:
    @st.fragment
    def _render_bwaware_best_tab():
        st.markdown("""
        <div class="dashboard-hero">
            <div class="dashboard-hero__eyebrow">Bandwidth-Aware · Greedy Descent</div>
            <h1>Bandwidth-Aware Greedy-Descent Results</h1>
            <p>Only bandwidth-aware quantization runs produced by the greedy weight-format descent are shown here.</p>
        </div>
        """, unsafe_allow_html=True)

        @st.cache_data(ttl=30, show_spinner=False)
        def _load_bwaware_greedy_results(project_root):
            from runspace.src.database.bwaware_results import load_greedy_descent_results

            return load_greedy_descent_results(project_root)

        runs_to_display = _load_bwaware_greedy_results(PROJECT_ROOT)

        if not runs_to_display:
            st.info(
                "No greedy-descent bandwidth-aware quant results found yet. "
                "Run **BW-Aware Quant** from **Run Models** with a greedy-descent experiment variant."
            )
            return

        labels = [run["label"] for run in runs_to_display]
        selected_label = st.selectbox(
            "Bandwidth-aware result",
            labels,
            index=0,
            key="bwaware_greedy_result_select",
        )
        run = runs_to_display[labels.index(selected_label)]

        data = run["data"]
        ref = data.get("ref_fp32", {}) or {}
        rows = _bandwidth_aware_quant_rows(data)
        points_df = pd.DataFrame(rows)

        st.caption(run["path"])
        st.success("Greedy descent result")
        m1, m2, m3 = st.columns(3)
        m1.metric("Model", data.get("model_name", "N/A"))
        m2.metric("FP32 Acc1", f"{float(ref.get('accuracy', 0.0)):.3f}%")
        m3.metric("Sweep Points", len(points_df))

        # Show best weight formats if available
        best_weight_map_by_bits = data.get("best_weight_map_by_bits")
        if best_weight_map_by_bits and isinstance(best_weight_map_by_bits, dict):
            st.markdown("---")
            st.markdown("#### Best Weight Formats (per bit-width)")
            st.caption("Per-layer best weight format for each bit-width, loaded from the weight_quant_optimized CSV results.")

            # Build a summary of all unique formats across all layers & bit-widths
            all_fmts = set()
            layer_rows = []
            for layer, bits_map in sorted(best_weight_map_by_bits.items()):
                row = {"Layer": layer}
                for bits, fmt in sorted(bits_map.items()):
                    row[f"{bits}b"] = fmt
                    all_fmts.add(fmt)
                layer_rows.append(row)

            # Show format distribution as metrics
            fmt_counts = {}
            for layer, bits_map in best_weight_map_by_bits.items():
                for bits, fmt in bits_map.items():
                    fmt_counts[fmt] = fmt_counts.get(fmt, 0) + 1
            if fmt_counts:
                sorted_fmts = sorted(fmt_counts.items(), key=lambda x: -x[1])
                cols = st.columns(min(len(sorted_fmts), 6))
                for idx, (fmt, cnt) in enumerate(sorted_fmts):
                    cols[idx % len(cols)].metric(fmt, f"{cnt} entries")

            # Show layer -> bit-width -> format table
            layer_df = pd.DataFrame(layer_rows)
            # Reorder columns so Layer is first, then bit-width columns sorted
            bit_cols = sorted([c for c in layer_df.columns if c != "Layer"], key=lambda x: int(x.replace("b", "")))
            layer_df = layer_df[["Layer"] + bit_cols]
            st.dataframe(
                layer_df,
                width='stretch',
                hide_index=True,
                column_config={
                    "Layer": st.column_config.TextColumn("Layer", width="large"),
                },
            )

            # Show as JSON for copy-paste
            with st.expander("Show best_weight_map_by_bits JSON", expanded=False):
                st.json(best_weight_map_by_bits)
        elif data.get("best_weight_map") and isinstance(data.get("best_weight_map"), dict):
            # Legacy fallback for older runs
            st.markdown("---")
            st.markdown("#### Best Weight Formats (legacy)")
            st.caption("Per-layer best weight formats (legacy overall-best map).")
            st.json(data.get("best_weight_map"))

        if not points_df.empty:
            sort_cols = [c for c in ["min_bits", "cache_size_M", "b"] if c in points_df.columns]
            points_df = points_df.sort_values(sort_cols)
            st.markdown("##### Accuracy vs Speedup")
            _render_bandwidth_aware_quant_chart(data, points_df)

            hidden_cols = {"min_bits", "norm_speedup", "cache_label"}
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

    _render_bwaware_best_tab()

# ── Architecture Graph Tab ───────────────────────────────────────────────────
