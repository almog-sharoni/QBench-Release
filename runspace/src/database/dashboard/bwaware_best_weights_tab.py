with tab_bwaware_best:
    @st.fragment
    def _render_bwaware_best_tab():
        st.markdown("""
        <div class="dashboard-hero">
            <div class="dashboard-hero__eyebrow">Bandwidth-Aware · Best Weights</div>
            <h1>Bandwidth-Aware Result — Best Weights</h1>
            <p>Bandwidth-aware quantization results that use the per-layer best weight formats from the latest <code>weight_quant_optimized</code> DB run instead of SIGNED_FORMATS_BY_BITS.</p>
        </div>
        """, unsafe_allow_html=True)

        # Load results from the best-weights output directory
        @st.cache_data(ttl=30, show_spinner=False)
        def _load_bwaware_best_results(project_root):
            results_roots = [
                (
                    "bandwidth_aware_quant_best_weights",
                    os.path.join(project_root, "runspace/experiments/bandwidth_aware_quant/results_best_weights"),
                ),
                (
                    "bandwidth_aware_quant",
                    os.path.join(project_root, "runspace/experiments/bandwidth_aware_quant/results"),
                ),
                (
                    "baselines_vs_dynamic_runs",
                    os.path.join(project_root, "runspace/experiments/baselines_vs_dynamic_runs/results/bandwidth_aware"),
                ),
            ]
            best_runs = []
            all_runs = []

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
                        all_runs.append({
                            "label": label,
                            "path": json_path,
                            "error": str(exc),
                        })
                        continue

                    run_info = {
                        "label": label,
                        "path": json_path,
                        "dir": dirpath,
                        "data": data,
                        "model_name": data.get("model_name", rel_dir),
                        "is_best_weights": "best_weights" in source_label or "best_weights" in rel_dir,
                    }
                    all_runs.append(run_info)
                    if run_info["is_best_weights"]:
                        best_runs.append(run_info)

            return sorted(best_runs, key=lambda r: r["label"]), sorted(all_runs, key=lambda r: r["label"])

        best_runs, all_runs = _load_bwaware_best_results(PROJECT_ROOT)

        if not best_runs:
            st.info(
                "No best-weights bandwidth-aware quant results found yet. "
                "Run the **BW-Aware Quant** experiment from the **Run Models** tab with **Use best weights from DB** enabled."
            )
            # Still show all runs if any exist, so the user can compare
            show_all = st.checkbox("Show all bandwidth-aware results", value=True, key="bwaware_best_show_all_fallback")
            if show_all and all_runs:
                runs_to_display = all_runs
            else:
                return
        else:
            show_all = st.checkbox("Show all bandwidth-aware results", value=False, key="bwaware_best_show_all")
            runs_to_display = all_runs if show_all else best_runs

        labels = [run["label"] for run in runs_to_display]
        selected_label = st.selectbox(
            "Bandwidth-aware result",
            labels,
            index=0,
            key="bwaware_best_result_select",
        )
        run = runs_to_display[labels.index(selected_label)]

        if run.get("error"):
            st.error(f"Could not parse {run['path']}: {run['error']}")
            return

        data = run["data"]
        ref = data.get("ref_fp32", {}) or {}
        rows = _bandwidth_aware_quant_rows(data)
        points_df = pd.DataFrame(rows)

        st.caption(run["path"])
        if run.get("is_best_weights"):
            st.success("This result uses best weight formats from the DB.")
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
