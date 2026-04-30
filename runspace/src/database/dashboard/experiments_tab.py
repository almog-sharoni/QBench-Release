st.set_page_config(
    page_title="QBench Experiment Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_global_styles()
import streamlit.components.v1 as _dashboard_components
_dashboard_components.html(
    """
    <script>
    (() => {
      const doc = window.parent.document;
      const key = "qbench-dashboard-sidebar-initial-collapse";
      if (window.sessionStorage.getItem(key) === "1") return;
      window.sessionStorage.setItem(key, "1");

      function isSidebarOpen() {
        const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
        if (!sidebar) return false;
        const rect = sidebar.getBoundingClientRect();
        return rect.width > 80 && rect.left > -rect.width + 20;
      }

      function collapseOnce() {
        if (!isSidebarOpen()) return;
        const button = doc.querySelector('[data-testid="stSidebarCollapseButton"] button')
          || doc.querySelector('[data-testid="stSidebarCollapseButton"]')
          || doc.querySelector('button[title*="Close sidebar"]')
          || doc.querySelector('button[aria-label*="Close sidebar"]');
        if (button) button.click();
      }

      setTimeout(collapseOnce, 120);
      setTimeout(collapseOnce, 500);
    })();
    </script>
    """,
    height=0,
)
flash_message = st.session_state.pop("dashboard_flash_message", None)
if flash_message:
    st.success(flash_message)


@st.dialog("Delete Database?", width="small")
def show_delete_database_dialog(db_name):
    st.warning(f"You are about to permanently delete `{db_name}`.")
    st.caption("This removes the database file from `runspace/database` and cannot be undone.")
    col_cancel, col_delete = st.columns(2)
    if col_cancel.button("Cancel", key=f"cancel_delete_db_{db_name}", width='stretch'):
        st.rerun()
    if col_delete.button("Yes, Delete DB", key=f"confirm_delete_db_modal_{db_name}", type="primary", width='stretch'):
        try:
            next_db_name = choose_db_after_delete(db_name)
            delete_database_file(db_name)
        except (FileNotFoundError, ValueError) as exc:
            st.error(str(exc))
        else:
            reset_filters_for_db_change()
            st.session_state["pending_selected_experiment_db"] = next_db_name
            st.session_state["dashboard_flash_message"] = f"Deleted `{db_name}`."
            st.rerun()

# Initial Session State for Presets
if 'presets' not in st.session_state:
    st.session_state.presets = load_presets()

st.sidebar.header("Database")
run_kind_options = {
    "Classification": "classification",
    "Feature Matching": "feature_matching",
}
run_kind_labels_by_value = {value: label for label, value in run_kind_options.items()}

def _selected_run_kind_from_query():
    try:
        raw_kind = st.query_params.get("run_type", "classification")
    except Exception:
        raw_kind = "classification"
    return raw_kind if raw_kind in run_kind_labels_by_value else "classification"

def _sync_run_kind_to_query():
    selected_label = st.session_state.get("selected_dashboard_run_kind_label", "Classification")
    selected_kind = run_kind_options.get(selected_label, "classification")
    try:
        st.query_params["run_type"] = selected_kind
    except Exception:
        pass
    reset_filters_for_db_change()

selected_run_kind_from_query = _selected_run_kind_from_query()
default_run_kind_label = run_kind_labels_by_value[selected_run_kind_from_query]
if (
    "selected_dashboard_run_kind_label" not in st.session_state
    or run_kind_options.get(st.session_state.get("selected_dashboard_run_kind_label")) != selected_run_kind_from_query
):
    st.session_state["selected_dashboard_run_kind_label"] = default_run_kind_label
selected_run_kind_label = st.sidebar.radio(
    "Run type",
    options=list(run_kind_options.keys()),
    index=list(run_kind_options.keys()).index(default_run_kind_label),
    key="selected_dashboard_run_kind_label",
    on_change=_sync_run_kind_to_query,
)
DASHBOARD_RUN_KIND = run_kind_options[selected_run_kind_label]
try:
    st.query_params["run_type"] = DASHBOARD_RUN_KIND
except Exception:
    pass
db_options = list_database_files_for_run_kind(DASHBOARD_RUN_KIND)
default_db_name = (
    os.path.basename(FM_DB_PATH)
    if DASHBOARD_RUN_KIND == "feature_matching"
    else os.path.basename(DEFAULT_DB_PATH)
)
pending_db_selection = st.session_state.pop("pending_selected_experiment_db", None)
if pending_db_selection in db_options:
    st.session_state[f"selected_experiment_db_{DASHBOARD_RUN_KIND}"] = pending_db_selection
default_db_index = db_options.index(default_db_name) if default_db_name in db_options else 0
selected_db_name = st.sidebar.selectbox(
    "Experiment DB",
    options=db_options,
    index=default_db_index,
    key=f"selected_experiment_db_{DASHBOARD_RUN_KIND}",
    on_change=reset_filters_for_db_change,
    help="Choose a SQLite database from runspace/database for the selected run type.",
)
DB_PATH = os.path.join(DB_FOLDER, selected_db_name)
FM_DB_PATH = DB_PATH if DASHBOARD_RUN_KIND == "feature_matching" else os.path.join(DB_FOLDER, "fm_runs.db")
st.sidebar.caption(f"Using `{selected_db_name}`")

with st.sidebar.expander("Manage DB", expanded=False):
    rename_db_name = st.text_input(
        "Rename selected DB",
        value=selected_db_name,
        key=f"rename_selected_db_name_{selected_db_name}",
        help="Renames the selected database file inside runspace/database.",
    )
    rename_safe_name = make_safe_db_filename(rename_db_name)
    st.caption(f"New filename: `{rename_safe_name}`")
    if st.button("Rename DB", key="rename_selected_db_btn", width='stretch'):
        try:
            renamed_db_name = rename_database_file(selected_db_name, rename_db_name)
        except (FileExistsError, FileNotFoundError, ValueError) as exc:
            st.error(str(exc))
        else:
            reset_filters_for_db_change()
            st.session_state["pending_selected_experiment_db"] = renamed_db_name
            st.session_state["dashboard_flash_message"] = (
                f"Renamed `{selected_db_name}` to `{renamed_db_name}`."
            )
            st.rerun()

    st.markdown("---")
    confirm_delete_db = st.checkbox(
        f"Confirm deleting `{selected_db_name}`",
        key=f"confirm_delete_selected_db_{selected_db_name}",
    )
    if st.button(
        "Delete Selected DB",
        key="delete_selected_db_btn",
        width='stretch',
        disabled=not confirm_delete_db,
    ):
        show_delete_database_dialog(selected_db_name)

st.sidebar.header("⚡ Performance")
run_window_label = st.sidebar.selectbox(
    "Rows to Load",
    options=list(RUN_WINDOW_TO_LIMIT.keys()),
    index=1,
    help="Loads only the newest N runs to keep reruns fast. Use 'All' when you need full history.",
)
selected_run_limit = RUN_WINDOW_TO_LIMIT[run_window_label]
load_graphs_on_demand = st.sidebar.checkbox(
    "Generate Graphs On Demand",
    value=True,
    help="Generates architecture graphs only when requested. Graphs are not cached.",
)
use_cached_graphs = st.sidebar.checkbox(
    "Use Cached Architecture Graphs",
    value=True,
    help="Loads previously generated architecture graphs from the selected database. Use Regenerate Graph when code changes.",
)
st.sidebar.caption("Data updates on Streamlit reruns. Active run logs refresh in the Run Models tab without reloading the whole page.")

st.markdown("---")

st.markdown("---")

tab_exp, tab_cache, tab_runner, tab_graph = st.tabs(["📊 Experiments", "🗄️ Cache Simulation", "🚀 Run Models", "🏗️ Architecture Graph"])

import streamlit.components.v1 as _dashboard_components
_dashboard_components.html(
    """
    <script>
    (() => {
      const labels = ["📊 Experiments", "🗄️ Cache Simulation", "🚀 Run Models", "🏗️ Architecture Graph"];
      const storageKey = "qbench-dashboard-active-tab";
      const doc = window.parent.document;

      function textOf(el) {
        return (el && el.innerText || "").replace(/\\s+/g, " ").trim();
      }

      function mainTabs() {
        return Array.from(doc.querySelectorAll('[role="tab"]'))
          .filter(tab => labels.includes(textOf(tab)));
      }

      function restore() {
        const wanted = window.localStorage.getItem(storageKey);
        if (!wanted) return;
        const tab = mainTabs().find(t => textOf(t) === wanted);
        if (tab && tab.getAttribute("aria-selected") !== "true") {
          tab.click();
        }
      }

      function install() {
        mainTabs().forEach(tab => {
          if (tab.dataset.qbenchRememberTab === "1") return;
          tab.dataset.qbenchRememberTab = "1";
          tab.addEventListener("click", () => {
            window.localStorage.setItem(storageKey, textOf(tab));
          });
        });
      }

      setTimeout(restore, 120);
      setTimeout(restore, 500);
      install();
      new MutationObserver(install).observe(doc.body, {childList: true, subtree: true});
    })();
    </script>
    """,
    height=0,
)

# Renderer exposed by the else: block below; None if db is empty.
_graph_renderer_fn = None

# Enter the Experiments tab context without re-indenting the existing block.
tab_exp.__enter__()

with st.spinner("Loading experiment runs..."):
    df = get_fm_runs(selected_run_limit) if DASHBOARD_RUN_KIND == "feature_matching" else get_runs(selected_run_limit)

if df.empty:
    st.warning("No runs found in the database yet. Run an experiment first!")
elif DASHBOARD_RUN_KIND == "feature_matching":
    st.markdown("""
    <div class="dashboard-hero">
        <div class="dashboard-hero__eyebrow">Feature Matching · Runs</div>
        <h1>Feature Matching Experiments</h1>
        <p>Scan matching metrics, pose AUC, and reference deltas from the selected feature-matching database.</p>
    </div>
    """, unsafe_allow_html=True)
    st.caption(f"Showing {len(df)} feature-matching runs from `{os.path.basename(DB_PATH)}`.")

    fm_models = sorted(df['model_name'].dropna().unique().tolist()) if 'model_name' in df.columns else []
    fm_formats = _sort_quant_formats(df['weight_dt'].dropna().unique().tolist()) if 'weight_dt' in df.columns else []
    fm_statuses = sorted(df['status'].dropna().unique().tolist()) if 'status' in df.columns else []
    c1, c2, c3 = st.columns(3)
    sel_models = c1.multiselect("Model", fm_models, default=fm_models, key="fm_main_model_filter")
    sel_formats = c2.multiselect("Weight format", fm_formats, default=fm_formats, key="fm_main_fmt_filter")
    sel_statuses = c3.multiselect("Status", fm_statuses, default=fm_statuses, key="fm_main_status_filter")

    fm_view = df.copy()
    if sel_models and 'model_name' in fm_view.columns:
        fm_view = fm_view[fm_view['model_name'].isin(sel_models)]
    if sel_formats and 'weight_dt' in fm_view.columns:
        fm_view = fm_view[fm_view['weight_dt'].isin(sel_formats)]
    if sel_statuses and 'status' in fm_view.columns:
        fm_view = fm_view[fm_view['status'].isin(sel_statuses)]

    for metric in ['matching_precision', 'matching_score', 'mean_num_matches',
                   'pose_auc_5', 'pose_auc_10', 'pose_auc_20']:
        ref_col = f'ref_{metric}'
        if metric in fm_view.columns and ref_col in fm_view.columns:
            fm_view[f'Δ_{metric}'] = pd.to_numeric(fm_view[metric], errors='coerce') \
                - pd.to_numeric(fm_view[ref_col], errors='coerce')

    display_cols = [c for c in [
        'id', 'model_name', 'weight_dt', 'activation_dt', 'experiment_type',
        'status', 'run_date',
        'matching_precision', 'ref_matching_precision', 'Δ_matching_precision',
        'matching_score', 'ref_matching_score', 'Δ_matching_score',
        'mean_num_matches', 'ref_mean_num_matches', 'Δ_mean_num_matches',
        'pose_auc_5', 'ref_pose_auc_5', 'Δ_pose_auc_5',
        'pose_auc_10', 'ref_pose_auc_10', 'Δ_pose_auc_10',
        'pose_auc_20', 'ref_pose_auc_20', 'Δ_pose_auc_20',
        'fm_num_keypoints', 'fm_mean_score', 'fm_desc_norm', 'fm_repeatability',
    ] if c in fm_view.columns]
    st.dataframe(fm_view[display_cols], width='stretch', hide_index=True)

    with st.expander("Show config_json for filtered rows", expanded=False):
        for _, row in fm_view.iterrows():
            st.markdown(f"**[{row['id']}] {row['model_name']} - {row['weight_dt']}/{row['activation_dt']}**")
            st.json(_safe_json_load(row.get('config_json')))
else:
    with st.spinner("Preparing dashboard data..."):
        df = preprocess_runs_df(df)
    @st.dialog("🔬 Layer Quantization Breakdown", width="large")
    def show_layer_breakdown(run_row):
        model = run_row.get('model_name', '')
        w_dt  = run_row.get('weight_dt', '')
        a_dt  = run_row.get('activation_dt', '')
        exp_t = run_row.get('experiment_type', '')
        weight_map_json, input_map_json, is_input_exp = _resolve_maps_for_display(run_row)

        st.markdown(f"**Model:** `{model}`  |  **Weight DT:** `{w_dt}`  |  **Activation DT:** `{a_dt}`")
        st.markdown(f"**Experiment:** `{exp_t}`")
        st.markdown("---")

        def _render_map_section(title, raw_json, mode_label):
            if not raw_json or (isinstance(raw_json, float) and pd.isna(raw_json)):
                return
            try:
                quant_map = json.loads(raw_json)
            except Exception:
                st.error(f"Could not parse {title} map JSON.")
                return

            def _format_count_summary(counts):
                if not isinstance(counts, dict) or not counts:
                    return None
                items = []
                sortable_counts = []
                for fmt, cnt in counts.items():
                    try:
                        cnt_i = int(cnt)
                    except Exception:
                        continue
                    sortable_counts.append((str(fmt), cnt_i))
                for fmt, cnt_i in sorted(sortable_counts, key=lambda x: (-x[1], x[0])):
                    if cnt_i > 0:
                        items.append(f"{fmt}×{cnt_i}")
                return ", ".join(items) if items else None

            rows = []
            chunk_dist_rows = []  # For per-layer chunk distribution (dynamic runs)
            for layer, value in quant_map.items():
                # New format: {"format": "fp4_e1m2", "type": "Conv2d", ...}
                # Old format: "fp4_e1m2"  or  ["fp4_e1m2", ...]
                if isinstance(value, dict):
                    fmt = value.get("format", "?")
                    layer_type = value.get("type", "?")
                    per_chunk_counts = value.get("format_counts")
                    total_chunks = value.get("total_chunks")
                    dominant_fmt = value.get("dominant_format")
                else:
                    fmt = value
                    layer_type = "?"
                    per_chunk_counts = None
                    total_chunks = None
                    dominant_fmt = None

                count_summary = _format_count_summary(per_chunk_counts)
                if count_summary and len(per_chunk_counts) > 1:
                    rows.append({
                        "Layer": layer,
                        "Type": layer_type,
                        "Format": count_summary,
                        "Dominant": str(dominant_fmt or "?"),
                        "Mode": "per-chunk",
                    })
                elif isinstance(fmt, list):
                    counts = {}
                    for f in fmt:
                        counts[f] = counts.get(f, 0) + 1
                    fmt_str = ", ".join(
                        f"{f}×{c}" for f, c in sorted(counts.items(), key=lambda x: -x[1])
                    )
                    rows.append({
                        "Layer": layer,
                        "Type": layer_type,
                        "Format": fmt_str,
                        "Dominant": str(dominant_fmt or "?"),
                        "Mode": "per-chunk",
                    })
                else:
                    rows.append({
                        "Layer": layer,
                        "Type": layer_type,
                        "Format": count_summary or str(fmt),
                        "Dominant": str(dominant_fmt or fmt),
                        "Mode": mode_label,
                    })

                if per_chunk_counts and total_chunks:
                    row = {"Layer": layer, "Type": layer_type, "Total Chunks": total_chunks}
                    row.update({f: per_chunk_counts.get(f, 0) for f in per_chunk_counts})
                    chunk_dist_rows.append(row)

            if not rows:
                return

            breakdown_df = pd.DataFrame(rows)
            format_counts = breakdown_df["Format"].value_counts()

            st.markdown(f"#### {title} — {len(rows)} layers")
            cols = st.columns(min(len(format_counts), 6))
            for idx, (fmt, cnt) in enumerate(format_counts.items()):
                cols[idx % len(cols)].metric(fmt, f"{cnt} layers")
            st.dataframe(
                breakdown_df,
                width='stretch',
                column_config={
                    "Layer":  st.column_config.TextColumn("Layer", width="large"),
                    "Type":   st.column_config.TextColumn("Type"),
                    "Format": st.column_config.TextColumn("Format"),
                    "Dominant": st.column_config.TextColumn("Dominant"),
                    "Mode":   st.column_config.TextColumn("Mode"),
                },
            )

            if chunk_dist_rows:
                with st.expander("📊 Chunk Format Distribution (per layer)", expanded=False):
                    chunk_df = pd.DataFrame(chunk_dist_rows).fillna(0)
                    # Move Layer/Type/Total Chunks to front; format columns sorted by total usage
                    fixed_cols = ["Layer", "Type", "Total Chunks"]
                    fmt_cols = sorted(
                        [c for c in chunk_df.columns if c not in fixed_cols],
                        key=lambda c: -chunk_df[c].sum()
                    )
                    chunk_df = chunk_df[fixed_cols + fmt_cols]
                    for fc in fmt_cols:
                        chunk_df[fc] = chunk_df[fc].astype(int)
                    chunk_df["Total Chunks"] = chunk_df["Total Chunks"].astype(int)
                    st.dataframe(chunk_df, width='stretch')

        act_mode = "dynamic" if ("dyn" in a_dt.lower() or "oracle" in a_dt.lower()) else "static"
        if not is_input_exp:
            _render_map_section("⚖️ Weight Formats", weight_map_json, "per-layer")
        _render_map_section("⚡ Input Formats",  input_map_json,  act_mode)

    @st.dialog("⚙️ Run Configurations", width="large")
    def show_run_config(run_rows):
        """run_rows: list of row dicts."""
        if len(run_rows) == 1:
            row = run_rows[0]
            weight_map_json, input_map_json, is_input_exp = _resolve_maps_for_display(row)
            st.markdown(f"**Model:** `{row.get('model_name','')}` | **Exp:** `{row.get('experiment_type','')}` | `{row.get('weight_dt','')}` / `{row.get('activation_dt','')}`")
            st.markdown("---")
            cfg_tab, win_tab = st.tabs(["Config", "Win Rates"])

            with cfg_tab:
                raw = row.get('config_json')
                if not raw or (isinstance(raw, float) and pd.isna(raw)):
                    st.info("No config stored for this run.")
                else:
                    try:
                        st.json(json.loads(raw), expanded=True)
                    except Exception:
                        st.code(str(raw), language="json")

            with win_tab:
                if not is_input_exp:
                    _render_weight_win_rates(weight_map_json)
                    if _safe_json_load(weight_map_json) is not None and _safe_json_load(input_map_json) is not None:
                        st.markdown("---")
                _render_input_win_rates(input_map_json)
        else:
            tab_labels = [
                f"{r.get('model_name','')} · {r.get('weight_dt','')} / {r.get('activation_dt','')}"
                for r in run_rows
            ]
            tabs = st.tabs(tab_labels)
            for tab, row in zip(tabs, run_rows):
                with tab:
                    weight_map_json, input_map_json, is_input_exp = _resolve_maps_for_display(row)
                    st.markdown(f"**Experiment:** `{row.get('experiment_type','')}`")
                    cfg_tab, win_tab = st.tabs(["Config", "Win Rates"])

                    with cfg_tab:
                        raw = row.get('config_json')
                        if not raw or (isinstance(raw, float) and pd.isna(raw)):
                            st.info("No config stored for this run.")
                        else:
                            try:
                                st.json(json.loads(raw), expanded=True)
                            except Exception:
                                st.code(str(raw), language="json")

                    with win_tab:
                        if not is_input_exp:
                            _render_weight_win_rates(weight_map_json)
                            if _safe_json_load(weight_map_json) is not None and _safe_json_load(input_map_json) is not None:
                                st.markdown("---")
                        _render_input_win_rates(input_map_json)

    @st.dialog("🗑️ Delete Runs From Database", width="large")
    def show_delete_runs_dialog(run_rows, table_index):
        if not run_rows:
            st.info("No runs selected.")
            return

        selected_df = pd.DataFrame(run_rows)
        run_ids = []
        for row in run_rows:
            run_id = row.get('id')
            if pd.notna(run_id):
                run_ids.append(int(run_id))
        run_ids = sorted(set(run_ids))

        st.warning(f"This permanently deletes the selected rows from `{selected_db_name}`. This action cannot be undone.")

        preview_cols = [
            'id', 'model_name', 'experiment_type', 'weight_dt', 'activation_dt',
            'acc1', 'acc5', 'status', 'run_date'
        ]
        existing_preview_cols = [c for c in preview_cols if c in selected_df.columns]
        st.dataframe(
            selected_df[existing_preview_cols],
            width='stretch',
            hide_index=True,
            column_config={
                "id": "DB ID",
                "model_name": "Model",
                "experiment_type": "Exp. Type",
                "weight_dt": "Weight DT",
                "activation_dt": "Activation DT",
                "acc1": st.column_config.NumberColumn("Acc1 (%)", format="%.2f"),
                "acc5": st.column_config.NumberColumn("Acc5 (%)", format="%.2f"),
                "status": "Status",
                "run_date": "Date",
            },
        )

        st.caption(f"Selected rows: {len(run_ids)}")
        if st.button(
            f"🗑️ Permanently Delete {len(run_ids)} Selected Row{'s' if len(run_ids) != 1 else ''}",
            key=f"confirm_delete_runs_{table_index}",
            type="primary",
            width='stretch',
        ):
            deleted_count = delete_runs_by_ids(run_ids)
            st.session_state["dashboard_flash_message"] = (
                f"Deleted {deleted_count} run{'s' if deleted_count != 1 else ''} from the database."
            )
            for key in (
                f"table_{table_index}",
            ):
                st.session_state.pop(key, None)
            st.rerun()

    @st.dialog("✏️ Edit Experiment Name", width="large")
    def show_edit_experiment_dialog(run_rows, table_index):
        if not run_rows:
            st.info("No runs selected.")
            return

        selected_df = pd.DataFrame(run_rows)
        run_ids = []
        for row in run_rows:
            run_id = row.get('id')
            if pd.notna(run_id):
                run_ids.append(int(run_id))
        run_ids = sorted(set(run_ids))

        current_values = sorted(
            selected_df.get('experiment_type', pd.Series(dtype=object)).fillna('').astype(str).unique().tolist()
        )
        default_value = current_values[0] if len(current_values) == 1 else ""
        if len(current_values) > 1:
            st.caption(f"Current experiment names: {', '.join(current_values)}")

        new_experiment_type = st.text_input(
            "Experiment name",
            value=default_value,
            key=f"edit_experiment_type_{table_index}",
            placeholder="e.g. input_quant_sweep_v2",
        )

        preview_cols = [
            'id', 'model_name', 'experiment_type', 'weight_dt', 'activation_dt',
            'acc1', 'acc5', 'status', 'run_date'
        ]
        existing_preview_cols = [c for c in preview_cols if c in selected_df.columns]
        st.dataframe(selected_df[existing_preview_cols], width='stretch', hide_index=True)

        if st.button(
            f"Save Experiment Name For {len(run_ids)} Row{'s' if len(run_ids) != 1 else ''}",
            key=f"confirm_edit_experiment_{table_index}",
            type="primary",
            width='stretch',
        ):
            if not str(new_experiment_type).strip():
                st.error("Enter an experiment name.")
                return
            updated_count = update_experiment_type_by_ids(run_ids, new_experiment_type)
            st.session_state["dashboard_flash_message"] = (
                f"Updated experiment name for {updated_count} row{'s' if updated_count != 1 else ''}."
            )
            st.session_state.pop(f"table_{table_index}", None)
            st.rerun()

    @st.dialog("💾 Create Database From Selected Rows", width="large")
    def show_create_db_dialog(run_rows, table_index):
        if not run_rows:
            st.info("No runs selected.")
            return

        selected_df = pd.DataFrame(run_rows)
        run_ids = []
        for row in run_rows:
            run_id = row.get('id')
            if pd.notna(run_id):
                run_ids.append(int(run_id))
        run_ids = sorted(set(run_ids))

        default_name = f"selected_runs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        db_name = st.text_input(
            "New DB filename",
            value=default_name,
            key=f"new_db_name_{table_index}",
            help="The database will be created in runspace/database.",
        )
        safe_name = make_safe_db_filename(db_name)
        destination_db_path = os.path.join(DB_FOLDER, safe_name)
        st.caption(f"Destination: `{destination_db_path}`")

        preview_cols = [
            'id', 'model_name', 'experiment_type', 'weight_dt', 'activation_dt',
            'acc1', 'acc5', 'status', 'run_date'
        ]
        existing_preview_cols = [c for c in preview_cols if c in selected_df.columns]
        st.dataframe(selected_df[existing_preview_cols], width='stretch', hide_index=True)

        if os.path.exists(destination_db_path):
            st.error("A database with this name already exists. Choose a new filename.")
            return

        if st.button(
            f"Create DB With {len(run_ids)} Row{'s' if len(run_ids) != 1 else ''}",
            key=f"confirm_create_db_{table_index}",
            type="primary",
            width='stretch',
        ):
            try:
                copied_count = create_database_from_run_ids(run_ids, destination_db_path)
            except FileExistsError as exc:
                st.error(str(exc))
                return
            st.session_state["dashboard_flash_message"] = (
                f"Created `{safe_name}` with {copied_count} selected row{'s' if copied_count != 1 else ''}."
            )
            st.session_state.pop(f"table_{table_index}", None)
            st.rerun()

    @st.dialog("📈 Accuracy Comparison", width="large")
    def show_large_chart(chart_df, selected_df):
        num_groups = chart_df['Label'].nunique()
        chart_width = max(1100, num_groups * 160)

        overview_tab, grouped_tab = st.tabs(["Overview", "Acc1 by Datatype"])

        with overview_tab:
            st.vega_lite_chart(chart_df, {
                'mark': {'type': 'bar', 'tooltip': True, 'size': 35},
                'config': {'view': {'stroke': 'transparent'}},
                'encoding': {
                    'x': {
                        'field': 'Label',
                        'type': 'nominal',
                        'axis': {
                            'labelAngle': -35,
                            'title': '',
                            'labelFontSize': 10,
                            'labelLineHeight': 12,
                            'labelOverlap': False,
                            'labelLimit': 220,
                            'labelExpr': "split(datum.label, '\\n')"
                        },
                        'sort': None
                    },
                    'y': {
                        'field': 'Accuracy (%)',
                        'type': 'quantitative',
                        'aggregate': 'mean',
                        'scale': {'zero': False, 'padding': 5, 'nice': True},
                        'title': 'Accuracy (%)'
                    },
                    'color': {
                        'field': 'MetricType',
                        'type': 'nominal',
                        'scale': {
                            'domain': ['Reference (Acc1)', 'Reference (Acc5)', 'Quantized (Acc1)', 'Quantized (Acc5)'],
                            'range': ['#00008b', '#add8e6', '#ff4500', '#ffa07a']
                        },
                        'legend': {'orient': 'top', 'align': 'left', 'padding': 10}
                    },
                    'xOffset': {
                        'field': 'MetricName',
                        'scale': {'paddingInner': 0}
                    }
                },
                'height': 600,
                'width': chart_width
            }, width='content')
            st.info("💡 Tip: Use the horizontal scrollbar above to see all models. Click '...' to save.")

        with grouped_tab:
            grouped_df = selected_df.copy()
            grouped_df['acc1'] = pd.to_numeric(grouped_df['acc1'], errors='coerce')
            grouped_df = grouped_df.dropna(subset=['acc1'])
            if grouped_df.empty:
                st.info("No Acc1 values are available for the selected rows.")
            else:
                grouped_df['Datatype'] = (
                    grouped_df['weight_dt'].astype(str) + "/" +
                    grouped_df['activation_dt'].astype(str)
                )
                if grouped_df['model_name'].nunique() > 1:
                    grouped_df['Group'] = grouped_df['model_name'].astype(str) + "\n" + grouped_df['Datatype']
                else:
                    grouped_df['Group'] = grouped_df['Datatype']
                grouped_df['Experiment Type'] = grouped_df['experiment_type'].fillna('unknown').astype(str)
                grouped_df['Acc1 (%)'] = grouped_df['acc1']

                grouped_width = max(
                    700,
                    grouped_df['Group'].nunique() * max(80, grouped_df['Experiment Type'].nunique() * 30),
                )
                st.vega_lite_chart(grouped_df, {
                    'mark': {'type': 'bar', 'tooltip': True, 'width': {'band': 1}},
                    'config': {'view': {'stroke': 'transparent'}},
                    'encoding': {
                        'x': {
                            'field': 'Group',
                            'type': 'nominal',
                            'scale': {'paddingInner': 0.35, 'paddingOuter': 0.08},
                            'axis': {
                                'labelAngle': -35,
                                'title': 'Datatype',
                                'labelFontSize': 10,
                                'labelLineHeight': 12,
                                'labelOverlap': False,
                                'labelLimit': 220,
                                'labelExpr': "split(datum.label, '\\n')",
                            },
                            'sort': None,
                        },
                        'xOffset': {
                            'field': 'Experiment Type',
                            'sort': None,
                            'scale': {'paddingInner': 0.0, 'paddingOuter': 0.0},
                        },
                        'y': {
                            'field': 'Acc1 (%)',
                            'type': 'quantitative',
                            'scale': {'zero': False, 'padding': 5, 'nice': True},
                            'title': 'Acc1 (%)',
                        },
                        'color': {
                            'field': 'Experiment Type',
                            'type': 'nominal',
                            'legend': {
                                'orient': 'top',
                                'align': 'left',
                                'padding': 10,
                                'labelLimit': 0,
                                'symbolLimit': 0,
                                'columns': 1,
                            },
                        },
                        'tooltip': [
                            {'field': 'model_name', 'title': 'Model'},
                            {'field': 'Experiment Type', 'title': 'Experiment'},
                            {'field': 'Datatype', 'title': 'Datatype'},
                            {'field': 'Acc1 (%)', 'title': 'Acc1 (%)', 'format': '.3f'},
                        ],
                    },
                    'height': 600,
                    'width': grouped_width,
                }, width='content')
                st.dataframe(
                    grouped_df[['Experiment Type']].drop_duplicates().sort_values('Experiment Type'),
                    width='stretch',
                    hide_index=True,
                )

    def render_full_graph_viewer(selected_model, graph_json, graph_meta, download_key):
            # Display metadata
            meta_col1, meta_col2, meta_col3 = st.columns(3)
            if graph_meta:
                meta_col1.metric("JSON Size", f"{graph_meta.get('graph_size_original', 0)/1024:.1f} KB")
                meta_col2.metric("Nodes", f"{graph_meta.get('num_nodes', 0)}")
                meta_col3.metric("Quantized Nodes", f"{graph_meta.get('num_quantized_layers', 0)}")
                generated_at = graph_meta.get('generated_at')
                if generated_at:
                    st.caption(f"Generated live at {generated_at}")
                generated_date = graph_meta.get('generated_date')
                if graph_meta.get('source') == 'cache' and generated_date:
                    st.caption(f"Loaded cached graph generated at {generated_date}")
        
            st.markdown("**Green** = Quantized Layers ")
            st.info("💡 **Interactive**: Click on dashed boxes (compound nodes) to collapse/expand them! Use mouse wheel to zoom.")
        
            import streamlit.components.v1 as components
        
            html_template = f"""
            <!DOCTYPE html>
            <html>
            <head>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.26.0/cytoscape.min.js"></script>
            <script src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.5.0/cytoscape-dagre.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/cytoscape-expand-collapse@4.1.0/cytoscape-expand-collapse.min.js"></script>
            <style>
                html, body {{
                    margin: 0;
                    padding: 0;
                    background: #f8fafc;
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                }}
                #graph-shell {{
                    position: relative;
                    width: 100%;
                    height: 810px;
                    border: 1px solid #e2e8f0;
                    border-radius: 10px;
                    background: #f8fafc;
                    overflow: hidden;
                }}
                #cy {{
                    position: absolute;
                    inset: 0;
                    z-index: 1;
                    background-color: #f8fafc;
                }}
                #toolbar {{
                    position: absolute;
                    top: 12px;
                    right: 12px;
                    z-index: 70;
                    display: flex;
                    gap: 8px;
                    flex-wrap: wrap;
                    justify-content: flex-end;
                    max-width: 90%;
                }}
                .graph-btn {{
                    cursor: pointer;
                    background: #ffffff;
                    border: 1px solid #bfdbfe;
                    border-radius: 8px;
                    color: #1d4ed8;
                    padding: 7px 10px;
                    font-size: 12px;
                    font-weight: 600;
                    box-shadow: 0 3px 10px rgba(0,0,0,0.12);
                    user-select: none;
                    transition: transform 0.08s ease, background-color 0.12s ease;
                }}
                .graph-btn:hover {{
                    background: #eff6ff;
                    transform: translateY(-1px);
                }}
                .graph-btn.active {{
                    background: #1d4ed8;
                    color: #ffffff;
                    border-color: #1e40af;
                }}
                #meta-layer {{
                    position: absolute;
                    inset: 0;
                    z-index: 45;
                    pointer-events: none;
                }}
                .meta-card {{
                    background: #ffffff;
                    border: 1px solid #cbd5e1;
                    border-radius: 8px;
                    padding: 7px 9px;
                    font-size: 11px;
                    color: #111827;
                    box-shadow: 0 6px 16px rgba(15, 23, 42, 0.18);
                    max-width: 260px;
                    line-height: 1.25;
                }}
                #hover-tooltip {{
                    position: absolute;
                    display: none;
                    pointer-events: none;
                }}
                .pinned-meta {{
                    position: absolute;
                    border-left: 3px solid #2563eb;
                    pointer-events: auto;
                    min-width: 180px;
                }}
                .meta-close {{
                    float: right;
                    border: none;
                    background: #e2e8f0;
                    color: #0f172a;
                    border-radius: 5px;
                    width: 20px;
                    height: 20px;
                    cursor: pointer;
                    font-weight: 700;
                    line-height: 18px;
                    padding: 0;
                    margin-left: 6px;
                }}
                .meta-close:hover {{
                    background: #cbd5e1;
                }}
                #all-meta-panel {{
                    position: absolute;
                    right: 12px;
                    top: 54px;
                    width: 340px;
                    max-height: 72%;
                    overflow: auto;
                    background: #ffffff;
                    border: 1px solid #93c5fd;
                    border-radius: 10px;
                    box-shadow: 0 8px 18px rgba(15, 23, 42, 0.2);
                    z-index: 65;
                    display: none;
                    padding: 10px;
                }}
                #all-meta-panel h4 {{
                    margin: 0 0 8px 0;
                    font-size: 13px;
                    color: #0f172a;
                }}
                #zoom-rect {{
                    position: absolute;
                    display: none;
                    border: 2px dashed #1d4ed8;
                    background: rgba(59, 130, 246, 0.18);
                    border-radius: 4px;
                    pointer-events: none;
                }}
            </style>
            </head>
            <body>
                <div id="graph-shell">
                    <div id="toolbar">
                        <button id="reset-view-btn" class="graph-btn">Reset View</button>
                        <button id="box-zoom-btn" class="graph-btn">Right-Drag Zoom: Off</button>
                        <button id="all-meta-btn" class="graph-btn">Show All Metadata</button>
                        <button id="fullscreen-btn" class="graph-btn">⛶ Fullscreen</button>
                    </div>
                    <div id="all-meta-panel"></div>
                    <div id="meta-layer">
                        <div id="hover-tooltip" class="meta-card"></div>
                        <div id="zoom-rect"></div>
                    </div>
                    <div id="cy"></div>
                </div>
                <script>
                    document.addEventListener('DOMContentLoaded', function() {{
                        var elements = {graph_json};
                        var ABS_ZOOM_MIN = 0.02;
                        var ZOOM_MAX = 3.0;
                        var FIT_PADDING = 80;
                        var currentMinZoom = ABS_ZOOM_MIN;

                        var cy = cytoscape({{
                            container: document.getElementById('cy'),
                            elements: elements,
                            minZoom: ABS_ZOOM_MIN,
                            maxZoom: ZOOM_MAX,
                            wheelSensitivity: 0.18,
                            boxSelectionEnabled: false,
                            style: [
                                {{
                                    selector: 'node',
                                    style: {{
                                        'content': 'data(label)',
                                        'text-valign': 'center',
                                        'text-halign': 'center',
                                        'background-color': 'data(color)',
                                        'border-width': 1,
                                        'border-color': '#94a3b8',
                                        'shape': 'round-rectangle',
                                        'width': 'label',
                                        'height': 'label',
                                        'padding': '10px',
                                        'font-size': '12px'
                                    }}
                                }},
                                {{
                                    selector: '$node > node',
                                    style: {{
                                        'background-color': '#e2e8f0',
                                        'padding-top': '25px',
                                        'padding-left': '10px',
                                        'padding-bottom': '10px',
                                        'padding-right': '10px',
                                        'text-valign': 'top',
                                        'text-halign': 'center',
                                        'border-color': '#94a3b8',
                                        'border-width': 2,
                                        'border-style': 'dashed',
                                        'font-size': '14px',
                                        'font-weight': 'bold'
                                    }}
                                }},
                                {{
                                    selector: 'edge',
                                    style: {{
                                        'curve-style': 'bezier',
                                        'target-arrow-shape': 'triangle',
                                        'line-color': '#cbd5e1',
                                        'target-arrow-color': '#cbd5e1',
                                        'width': 2
                                    }}
                                }},
                                {{
                                    selector: 'node.cy-expand-collapse-collapsed-node',
                                    style: {{
                                        'background-color': '#cbd5e1',
                                        'border-color': '#64748b',
                                        'border-width': 3,
                                        'shape': 'round-rectangle',
                                        'padding': '10px'
                                    }}
                                }}
                            ],
                            layout: {{
                                name: 'dagre',
                                rankDir: 'TB',
                                nodeSep: 50,
                                edgeSep: 10,
                                rankSep: 50
                            }}
                        }});

                        var api = cy.expandCollapse({{
                            layoutBy: {{
                                name: "dagre",
                                animate: true,
                                randomize: false,
                                fit: false
                            }},
                            fisheye: false,
                            animate: true,
                            undoable: false
                        }});

                        var cyContainer = document.getElementById('cy');
                        var hoverTooltip = document.getElementById('hover-tooltip');
                        var metaLayer = document.getElementById('meta-layer');
                        var allMetaPanel = document.getElementById('all-meta-panel');
                        var zoomRect = document.getElementById('zoom-rect');
                        var resetBtn = document.getElementById('reset-view-btn');
                        var allMetaBtn = document.getElementById('all-meta-btn');
                        var boxZoomBtn = document.getElementById('box-zoom-btn');
                        var fsBtn = document.getElementById('fullscreen-btn');

                        var pinnedCards = {{}};
                        var showAllMetadata = false;
                        var boxZoomEnabled = false;
                        var rightDragActive = false;
                        var rightDragStart = null;

                        function clamp(val, min, max) {{
                            return Math.max(min, Math.min(max, val));
                        }}

                        function computeFitZoom(paddingPx) {{
                            var bb = cy.elements().boundingBox();
                            if (!isFinite(bb.x1) || !isFinite(bb.x2) || !isFinite(bb.y1) || !isFinite(bb.y2)) {{
                                return currentMinZoom;
                            }}
                            var graphW = Math.max(bb.x2 - bb.x1, 1e-6);
                            var graphH = Math.max(bb.y2 - bb.y1, 1e-6);
                            var viewW = Math.max(cy.width() - (paddingPx * 2), 1);
                            var viewH = Math.max(cy.height() - (paddingPx * 2), 1);
                            var fitZoom = Math.min(viewW / graphW, viewH / graphH);
                            if (!isFinite(fitZoom) || fitZoom <= 0) {{
                                return currentMinZoom;
                            }}
                            return clamp(fitZoom, ABS_ZOOM_MIN, ZOOM_MAX);
                        }}

                        function refreshMinZoomToFitGraph() {{
                            currentMinZoom = computeFitZoom(FIT_PADDING);
                            cy.minZoom(currentMinZoom);
                            if (cy.zoom() < currentMinZoom) {{
                                cy.zoom(currentMinZoom);
                            }}
                        }}

                        function esc(val) {{
                            return String(val === undefined || val === null ? '-' : val)
                                .replace(/&/g, "&amp;")
                                .replace(/</g, "&lt;")
                                .replace(/>/g, "&gt;")
                                .replace(/"/g, "&quot;")
                                .replace(/'/g, "&#39;");
                        }}

                        function nodeHasMetadata(node) {{
                            return !!node.data('var_name');
                        }}

                        function metadataHtml(node) {{
                            var t = node.data('var_name') || node.data('label') || node.id();
                            var inS = node.data('input_shape') || '-';
                            var outS = node.data('output_shape') || '-';
                            var args = node.data('module_args') || '';
                            var argsHtml = args
                                ? "<div style='margin-top:4px;color:#475569;font-size:11px;'>" + esc(args) + "</div>"
                                : "";
                            return "<div>" +
                                "<div style='font-weight:700;color:#0f172a;'>" + esc(t) + "</div>" +
                                argsHtml +
                                "<div style='margin-top:6px;'><b>In:</b> " + esc(inS) + "</div>" +
                                "<div><b>Out:</b> " + esc(outS) + "</div>" +
                                "</div>";
                        }}

                        function getPointerPosition(evt) {{
                            var rect = cyContainer.getBoundingClientRect();
                            return {{
                                x: clamp(evt.clientX - rect.left, 0, cy.width()),
                                y: clamp(evt.clientY - rect.top, 0, cy.height())
                            }};
                        }}

                        function getMetaScale() {{
                            // Keep metadata tied to graph zoom, but clamp for readability.
                            return clamp(cy.zoom(), 0.62, 1.35);
                        }}

                        function applyCardScale(card, scale) {{
                            card.style.transform = "scale(" + scale + ")";
                            card.style.transformOrigin = "top left";
                        }}

                        function toNodeArray(collection) {{
                            if (!collection) return [];
                            if (collection.toArray) return collection.toArray();
                            return Array.isArray(collection) ? collection : [collection];
                        }}

                        function normalizeHoverNodes(nodes) {{
                            var filtered = nodes.filter(function(node) {{ return nodeHasMetadata(node); }});
                            if (!filtered.length) return [];

                            // Prefer concrete leaf modules over compound/group containers.
                            var leafNodes = filtered.filter(function(node) {{
                                return !(node.isParent && node.isParent());
                            }});
                            if (leafNodes.length) {{
                                filtered = leafNodes;
                            }}

                            // Deduplicate identical metadata blobs.
                            var seen = {{}};
                            var unique = [];
                            filtered.forEach(function(node) {{
                                var key = [
                                    node.data('var_name') || '',
                                    node.data('input_shape') || '',
                                    node.data('output_shape') || '',
                                    node.data('module_args') || ''
                                ].join('|');
                                if (!seen[key]) {{
                                    seen[key] = true;
                                    unique.push(node);
                                }}
                            }});

                            // Keep only the top-most relevant hit for hover.
                            return unique.length ? [unique[0]] : [];
                        }}

                        function sortByRenderedPosition(a, b) {{
                            var ap = a.renderedPosition();
                            var bp = b.renderedPosition();
                            var dy = ap.y - bp.y;
                            if (Math.abs(dy) > 1e-6) return dy;
                            return ap.x - bp.x;
                        }}

                        function buildLayerOrderMap() {{
                            var allNodes = toNodeArray(cy.nodes());
                            var byId = {{}};
                            var indegree = {{}};
                            var outgoing = {{}};

                            allNodes.forEach(function(node) {{
                                var id = node.id();
                                byId[id] = node;
                                indegree[id] = 0;
                                outgoing[id] = [];
                            }});

                            toNodeArray(cy.edges()).forEach(function(edge) {{
                                var src = edge.source().id();
                                var dst = edge.target().id();
                                if (src in byId && dst in byId) {{
                                    outgoing[src].push(dst);
                                    indegree[dst] += 1;
                                }}
                            }});

                            var queue = allNodes
                                .filter(function(node) {{ return indegree[node.id()] === 0; }})
                                .sort(sortByRenderedPosition);

                            var ordered = [];
                            while (queue.length > 0) {{
                                var node = queue.shift();
                                ordered.push(node);

                                outgoing[node.id()].forEach(function(dstId) {{
                                    indegree[dstId] -= 1;
                                    if (indegree[dstId] === 0) {{
                                        queue.push(byId[dstId]);
                                        queue.sort(sortByRenderedPosition);
                                    }}
                                }});
                            }}

                            if (ordered.length !== allNodes.length) {{
                                var seen = {{}};
                                ordered.forEach(function(node) {{ seen[node.id()] = true; }});
                                var leftovers = allNodes
                                    .filter(function(node) {{ return !seen[node.id()]; }})
                                    .sort(sortByRenderedPosition);
                                ordered = ordered.concat(leftovers);
                            }}

                            var orderMap = {{}};
                            ordered.forEach(function(node, idx) {{
                                orderMap[node.id()] = idx;
                            }});
                            return orderMap;
                        }}

                        function getNodesAtRenderedPoint(x, y) {{
                            if (typeof cy.elementsAtPoint === "function") {{
                                return normalizeHoverNodes(toNodeArray(
                                    cy.elementsAtPoint(x, y).filter(function(ele) {{
                                        return ele.isNode && ele.isNode() && nodeHasMetadata(ele);
                                    }})
                                ));
                            }}
                            return normalizeHoverNodes(toNodeArray(
                                cy.nodes().filter(function(node) {{
                                    if (!nodeHasMetadata(node)) return false;
                                    var bb = node.renderedBoundingBox();
                                    return x >= bb.x1 && x <= bb.x2 && y >= bb.y1 && y <= bb.y2;
                                }})
                            ));
                        }}

                        function renderHoverMetadata(evt) {{
                            if (showAllMetadata || rightDragActive) {{
                                hoverTooltip.style.display = "none";
                                return;
                            }}
                            var p = getPointerPosition(evt);
                            var nodes = getNodesAtRenderedPoint(p.x, p.y);
                            if (!nodes.length) {{
                                hoverTooltip.style.display = "none";
                                return;
                            }}

                            hoverTooltip.innerHTML = nodes.map(function(node, idx) {{
                                var divider = idx < nodes.length - 1
                                    ? "<div style='margin:8px 0;border-top:1px solid #e2e8f0;'></div>"
                                    : "";
                                return metadataHtml(node) + divider;
                            }}).join("");
                            var scale = getMetaScale() * 0.92;
                            applyCardScale(hoverTooltip, scale);
                            var cardW = hoverTooltip.offsetWidth * scale;
                            var cardH = hoverTooltip.offsetHeight * scale;
                            var bounds = nodes.reduce(function(acc, node) {{
                                var bb = node.renderedBoundingBox();
                                acc.x1 = Math.min(acc.x1, bb.x1);
                                acc.y1 = Math.min(acc.y1, bb.y1);
                                acc.x2 = Math.max(acc.x2, bb.x2);
                                return acc;
                            }}, {{ x1: Infinity, y1: Infinity, x2: -Infinity }});
                            var gap = 12;
                            var left = bounds.x2 + gap;
                            if (left + cardW > cy.width() - 6) {{
                                left = bounds.x1 - gap - cardW;
                            }}
                            var top = bounds.y1;
                            left = clamp(left, 6, cy.width() - cardW - 6);
                            top = clamp(top, 6, cy.height() - cardH - 6);
                            if (!isFinite(left) || !isFinite(top)) {{
                                left = clamp(p.x + 18, 6, cy.width() - cardW - 6);
                                top = clamp(p.y + 18, 6, cy.height() - cardH - 6);
                            }}
                            hoverTooltip.style.left = left + "px";
                            hoverTooltip.style.top = top + "px";
                            hoverTooltip.style.display = "block";
                        }}

                        function hideHoverMetadata() {{
                            hoverTooltip.style.display = "none";
                        }}

                        function removePinnedCard(nodeId) {{
                            if (!pinnedCards[nodeId]) return;
                            pinnedCards[nodeId].remove();
                            delete pinnedCards[nodeId];
                        }}

                        function positionPinnedCard(nodeId) {{
                            var card = pinnedCards[nodeId];
                            if (!card) return;
                            var node = cy.getElementById(nodeId);
                            if (!node || node.empty()) {{
                                removePinnedCard(nodeId);
                                return;
                            }}
                            if (!node.visible()) {{
                                card.style.display = "none";
                                return;
                            }}

                            card.style.display = "block";
                            var scale = getMetaScale();
                            applyCardScale(card, scale);
                            var cardW = card.offsetWidth * scale;
                            var cardH = card.offsetHeight * scale;
                            var bb = node.renderedBoundingBox();
                            var gap = 12;
                            var left = bb.x2 + gap;
                            if (left + cardW > cy.width() - 6) {{
                                left = bb.x1 - gap - cardW;
                            }}
                            var top = bb.y1;
                            left = clamp(left, 6, cy.width() - cardW - 6);
                            top = clamp(top, 6, cy.height() - cardH - 6);
                            card.style.left = left + "px";
                            card.style.top = top + "px";
                        }}

                        function updatePinnedCards() {{
                            Object.keys(pinnedCards).forEach(positionPinnedCard);
                        }}

                        function clearPinnedCards() {{
                            Object.keys(pinnedCards).forEach(removePinnedCard);
                        }}

                        function togglePinnedCard(node) {{
                            if (!nodeHasMetadata(node)) return;
                            var nodeId = node.id();
                            if (pinnedCards[nodeId]) {{
                                removePinnedCard(nodeId);
                                return;
                            }}

                            var card = document.createElement("div");
                            card.className = "meta-card pinned-meta";
                            card.innerHTML = "<button class='meta-close' title='Close'>×</button>" + metadataHtml(node);
                            card.querySelector(".meta-close").addEventListener("click", function(e) {{
                                e.stopPropagation();
                                removePinnedCard(nodeId);
                            }});
                            metaLayer.appendChild(card);
                            pinnedCards[nodeId] = card;
                            positionPinnedCard(nodeId);
                        }}

                        function renderAllMetadataPanel() {{
                            var layerOrder = buildLayerOrderMap();
                            var nodes = toNodeArray(
                                cy.nodes().filter(function(node) {{ return nodeHasMetadata(node); }})
                            );
                            nodes.sort(function(a, b) {{
                                var ai = (a.id() in layerOrder) ? layerOrder[a.id()] : Number.MAX_SAFE_INTEGER;
                                var bi = (b.id() in layerOrder) ? layerOrder[b.id()] : Number.MAX_SAFE_INTEGER;
                                if (ai !== bi) return ai - bi;
                                var av = a.data('var_name') || a.id();
                                var bv = b.data('var_name') || b.id();
                                return av.localeCompare(bv, undefined, {{ numeric: true, sensitivity: 'base' }});
                            }});
                            var rows = nodes.map(function(node, idx) {{
                                var divider = idx < nodes.length - 1
                                    ? "<div style='margin:8px 0;border-top:1px solid #e2e8f0;'></div>"
                                    : "";
                                return metadataHtml(node) + divider;
                            }}).join("");
                            allMetaPanel.innerHTML = "<h4>All Node Metadata (Layer Order, " + nodes.length + ")</h4>" + rows;
                        }}

                        function setShowAllMetadata(enabled) {{
                            showAllMetadata = enabled;
                            allMetaBtn.classList.toggle("active", enabled);
                            allMetaBtn.textContent = enabled ? "Hide All Metadata" : "Show All Metadata";
                            if (enabled) {{
                                hideHoverMetadata();
                                renderAllMetadataPanel();
                                allMetaPanel.style.display = "block";
                            }} else {{
                                allMetaPanel.style.display = "none";
                            }}
                        }}

                        function setBoxZoomEnabled(enabled) {{
                            boxZoomEnabled = enabled;
                            boxZoomBtn.classList.toggle("active", enabled);
                            boxZoomBtn.textContent = enabled ? "Right-Drag Zoom: On" : "Right-Drag Zoom: Off";
                        }}

                        function constrainPan() {{
                            refreshMinZoomToFitGraph();
                            var bb = cy.elements().boundingBox();
                            if (!isFinite(bb.x1) || !isFinite(bb.x2) || !isFinite(bb.y1) || !isFinite(bb.y2)) return;
                            var zoom = cy.zoom();
                            var pan = cy.pan();
                            var vw = cy.width();
                            var vh = cy.height();
                            var pad = 160;

                            var minX = vw - (bb.x2 * zoom) - pad;
                            var maxX = - (bb.x1 * zoom) + pad;
                            var minY = vh - (bb.y2 * zoom) - pad;
                            var maxY = - (bb.y1 * zoom) + pad;

                            var targetX = (minX > maxX) ? ((minX + maxX) / 2) : clamp(pan.x, minX, maxX);
                            var targetY = (minY > maxY) ? ((minY + maxY) / 2) : clamp(pan.y, minY, maxY);

                            if (targetX !== pan.x || targetY !== pan.y) {{
                                cy.pan({{ x: targetX, y: targetY }});
                            }}
                        }}

                        function resetView() {{
                            hideHoverMetadata();
                            setShowAllMetadata(false);
                            clearPinnedCards();
                            refreshMinZoomToFitGraph();
                            cy.fit(cy.elements(), FIT_PADDING);
                            cy.zoom(clamp(cy.zoom(), currentMinZoom, ZOOM_MAX));
                            constrainPan();
                            updatePinnedCards();
                        }}

                        function updateZoomRectangle(start, end) {{
                            var x = Math.min(start.x, end.x);
                            var y = Math.min(start.y, end.y);
                            var w = Math.abs(end.x - start.x);
                            var h = Math.abs(end.y - start.y);
                            zoomRect.style.left = x + "px";
                            zoomRect.style.top = y + "px";
                            zoomRect.style.width = w + "px";
                            zoomRect.style.height = h + "px";
                            zoomRect.style.display = "block";
                        }}

                        cyContainer.addEventListener("mousemove", renderHoverMetadata);
                        cyContainer.addEventListener("mouseleave", hideHoverMetadata);
                        cy.on("tap", "node", function(evt) {{
                            hideHoverMetadata();
                            togglePinnedCard(evt.target);
                        }});
                        cy.on("pan zoom resize render position", function() {{
                            constrainPan();
                            updatePinnedCards();
                        }});

                        allMetaBtn.addEventListener("click", function(e) {{
                            e.stopPropagation();
                            setShowAllMetadata(!showAllMetadata);
                        }});

                        boxZoomBtn.addEventListener("click", function(e) {{
                            e.stopPropagation();
                            setBoxZoomEnabled(!boxZoomEnabled);
                        }});

                        resetBtn.addEventListener("click", function(e) {{
                            e.stopPropagation();
                            resetView();
                        }});

                        cyContainer.addEventListener("contextmenu", function(e) {{
                            if (boxZoomEnabled || rightDragActive) {{
                                e.preventDefault();
                            }}
                        }});

                        cyContainer.addEventListener("mousedown", function(e) {{
                            if (!boxZoomEnabled || e.button !== 2) return;
                            e.preventDefault();
                            rightDragActive = true;
                            rightDragStart = getPointerPosition(e);
                            updateZoomRectangle(rightDragStart, rightDragStart);
                        }});

                        window.addEventListener("mousemove", function(e) {{
                            if (!rightDragActive || !rightDragStart) return;
                            var current = getPointerPosition(e);
                            updateZoomRectangle(rightDragStart, current);
                        }});

                        window.addEventListener("mouseup", function(e) {{
                            if (!rightDragActive || !rightDragStart) return;
                            var end = getPointerPosition(e);
                            var start = rightDragStart;
                            rightDragActive = false;
                            rightDragStart = null;
                            zoomRect.style.display = "none";

                            var w = Math.abs(end.x - start.x);
                            var h = Math.abs(end.y - start.y);
                            if (w < 12 || h < 12) return;

                            var pan = cy.pan();
                            var zoom = cy.zoom();
                            var x1 = (Math.min(start.x, end.x) - pan.x) / zoom;
                            var x2 = (Math.max(start.x, end.x) - pan.x) / zoom;
                            var y1 = (Math.min(start.y, end.y) - pan.y) / zoom;
                            var y2 = (Math.max(start.y, end.y) - pan.y) / zoom;

                            cy.animate({{
                                fit: {{ boundingBox: {{ x1: x1, y1: y1, x2: x2, y2: y2 }}, padding: 24 }},
                                duration: 220
                            }});
                            setTimeout(function() {{
                                constrainPan();
                                updatePinnedCards();
                            }}, 240);
                        }});

                        ['mousedown', 'mouseup', 'click', 'touchstart', 'touchend'].forEach(function(evt) {{
                            [fsBtn, resetBtn, allMetaBtn, boxZoomBtn].forEach(function(btn) {{
                                btn.addEventListener(evt, function(e) {{ e.stopPropagation(); }});
                            }});
                        }});

                        fsBtn.addEventListener('click', function(e) {{
                            e.stopPropagation();
                            var elem = document.getElementById('graph-shell');
                            if (elem.requestFullscreen) {{
                                elem.requestFullscreen();
                            }} else if (elem.webkitRequestFullscreen) {{
                                elem.webkitRequestFullscreen();
                            }} else if (elem.msRequestFullscreen) {{
                                elem.msRequestFullscreen();
                            }}
                        }});

                        // Smart auto-expansion: collapse compound nodes EXCEPT if it's the first appearance
                        cy.nodes('$node > node').sort((a, b) => b.neighborhood().length - a.neighborhood().length).forEach(function(node) {{
                            if (!node.data('is_first_appearance')) {{
                                api.collapse(node);
                            }}
                        }});

                        setBoxZoomEnabled(false);
                        setShowAllMetadata(false);
                        resetView();
                    }});
                </script>
            </body>
            </html>
            """
            components.html(html_template, height=820)
        
            # Download button
            st.download_button(
                label="⬇️ Download Graph JSON",
                data=graph_json,
                file_name=f"{selected_model}_architecture.json",
                mime="application/json",
                key=download_key
            )

    # Expose renderer for the Architecture Graph tab (defined here inside else:).
    _graph_renderer_fn = render_full_graph_viewer

    @st.dialog("🏗️ Architecture Graph Viewer", width="large")
    def show_graph_viewer(table_index, available_models):
        if not available_models:
            st.info("No models are available for this table.")
            return

        selected_model = st.selectbox(
            "Model",
            options=available_models,
            key=f"graph_model_dialog_{table_index}",
            help="Pick a model to generate its architecture graph live from the current code.",
        )

        graph_json = None
        graph_meta = None

        should_load_now = True
        force_regenerate = False
        if load_graphs_on_demand:
            btn_col1, btn_col2 = st.columns(2)
            if btn_col1.button("📊 Load Selected Graph", key=f"load_graph_dialog_{table_index}", type="primary", width='stretch'):
                should_load_now = True
            elif btn_col2.button("🔁 Regenerate", key=f"regen_graph_dialog_{table_index}", width='stretch'):
                should_load_now = True
                force_regenerate = True
            else:
                should_load_now = False

        if should_load_now:
            try:
                spinner_label = (
                    f"Regenerating architecture graph for {selected_model}..."
                    if force_regenerate or not use_cached_graphs
                    else f"Loading architecture graph for {selected_model}..."
                )
                with st.spinner(spinner_label):
                    graph_json, graph_meta, graph_source = load_or_generate_model_graph_bundle(
                        selected_model,
                        use_cache=use_cached_graphs,
                        force_regenerate=force_regenerate,
                    )
                    if graph_source == "cache":
                        st.success("Loaded cached graph.")
            except Exception as exc:
                st.error(f"Failed to load graph for `{selected_model}`: {exc}")
                return

        if load_graphs_on_demand and not should_load_now:
            st.info("Select a model and click `Load Selected Graph`. Use `Regenerate` when the model code changed.")
        elif graph_json:
            render_full_graph_viewer(
                selected_model=selected_model,
                graph_json=graph_json,
                graph_meta=graph_meta,
                download_key=f"dl_json_dialog_{table_index}",
            )
        else:
            st.info(f"No live architecture graph was generated for `{selected_model}`.")
    # --- Multi-Table Session State ---
    if 'num_tables' not in st.session_state:
        st.session_state.num_tables = 1

    def normalize_filter_selection(raw_value, options, default_to_all=True):
        options = list(options)
        option_set = set(options)
        if isinstance(raw_value, dict):
            normalized = [opt for opt, enabled in raw_value.items() if enabled and opt in option_set]
        elif isinstance(raw_value, (list, tuple, set)):
            normalized = [opt for opt in raw_value if opt in option_set]
        elif raw_value in option_set:
            normalized = [raw_value]
        else:
            normalized = []

        if default_to_all and raw_value is None:
            return options
        return normalized

    def _sync_filter_checkboxes(session_key, options, selected_values):
        selected_set = set(selected_values)
        for opt in options:
            st.session_state[f"{session_key}_choice_{repr(opt)}"] = opt in selected_set

    def render_filter_checklist(label, options, session_key, format_func=str, help_text=None):
        st.markdown(f"**{label}**")
        options = list(options)
        if not options:
            st.caption("No options available for the current data slice.")
            return []

        if session_key not in st.session_state:
            st.session_state[session_key] = options.copy()
        else:
            st.session_state[session_key] = normalize_filter_selection(
                st.session_state.get(session_key),
                options,
                default_to_all=False,
            )
        if st.session_state.pop(f"{session_key}_sync_required", False):
            _sync_filter_checkboxes(session_key, options, st.session_state[session_key])

        if help_text:
            st.caption(help_text)

        action_col1, action_col2, action_col3, info_col = st.columns([1, 1, 1, 1.4])
        if action_col1.button("Select all", key=f"{session_key}_select_all", width='stretch'):
            st.session_state[session_key] = options.copy()
            _sync_filter_checkboxes(session_key, options, st.session_state[session_key])
            st.rerun()
        if action_col2.button("Inverse", key=f"{session_key}_inverse", width='stretch'):
            selected_set = set(st.session_state[session_key])
            st.session_state[session_key] = [opt for opt in options if opt not in selected_set]
            _sync_filter_checkboxes(session_key, options, st.session_state[session_key])
            st.rerun()
        if action_col3.button("Clear", key=f"{session_key}_clear", width='stretch'):
            st.session_state[session_key] = []
            _sync_filter_checkboxes(session_key, options, st.session_state[session_key])
            st.rerun()

        selected_values = []
        for opt in options:
            widget_key = f"{session_key}_choice_{repr(opt)}"
            if widget_key not in st.session_state:
                st.session_state[widget_key] = opt in st.session_state[session_key]
            checked = st.checkbox(format_func(opt), key=widget_key)
            if checked:
                selected_values.append(opt)

        st.session_state[session_key] = selected_values
        info_col.caption(f"{len(selected_values)} of {len(options)} selected")
        return selected_values
    
    def add_table():
        st.session_state.num_tables += 1

    def remove_table(index):
        if st.session_state.num_tables > 1:
            # Shift session state for all tables above the removed index
            for i in range(index, st.session_state.num_tables - 1):
                keys_to_shift = [
                    'filter_m', 'filter_e', 'filter_wb', 'filter_we', 'filter_wm',
                    'filter_ab', 'filter_ae', 'filter_am', 'name_input'
                ]
                for prefix in keys_to_shift:
                    old_key = f"{prefix}_{i+1}"
                    new_key = f"{prefix}_{i}"
                    if old_key in st.session_state:
                        st.session_state[new_key] = st.session_state[old_key]
                    old_sync_key = f"{prefix}_{i+1}_sync_required"
                    new_sync_key = f"{prefix}_{i}_sync_required"
                    if old_sync_key in st.session_state:
                        st.session_state[new_sync_key] = st.session_state[old_sync_key]
                    for key in list(st.session_state.keys()):
                        old_choice_prefix = f"{prefix}_{i+1}_choice_"
                        if key.startswith(old_choice_prefix):
                            suffix = key[len(old_choice_prefix):]
                            st.session_state[f"{prefix}_{i}_choice_{suffix}"] = st.session_state[key]
        
            # Delete the last table's keys
            last_idx = st.session_state.num_tables - 1
            for prefix in ['filter_m', 'filter_e', 'filter_wb', 'filter_we', 'filter_wm', 'filter_ab', 'filter_ae', 'filter_am', 'name_input']:
                if f"{prefix}_{last_idx}" in st.session_state:
                    del st.session_state[f"{prefix}_{last_idx}"]
                if f"{prefix}_{last_idx}_sync_required" in st.session_state:
                    del st.session_state[f"{prefix}_{last_idx}_sync_required"]
                for key in list(st.session_state.keys()):
                    if key.startswith(f"{prefix}_{last_idx}_choice_"):
                        del st.session_state[key]
                
            st.session_state.num_tables -= 1
            st.rerun()

    # --- Global Filters & Settings ---
    st.sidebar.header("Global Filters & Settings")

    # Date Filter
    df['run_date_dt'] = pd.to_datetime(df['run_date'], errors='coerce')
    valid_run_dates = df['run_date_dt'].dropna()
    if valid_run_dates.empty:
        fallback_date = pd.Timestamp.utcnow().date()
        min_date = fallback_date
        max_date = fallback_date
    else:
        min_date = valid_run_dates.min().date()
        max_date = valid_run_dates.max().date()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="dashboard_date_range",
    )

    if len(date_range) == 2:
        start_date, end_date = date_range
        df = df[
            (df['run_date_dt'].dt.date >= start_date) & 
            (df['run_date_dt'].dt.date <= end_date)
        ]
    
    # Define options early for preset validation
    df['experiment_type'] = df['experiment_type'].fillna('unknown')
    df['model_name'] = df['model_name'].fillna('unknown')
    models = sorted(df['model_name'].unique())
    expr_types = sorted(df['experiment_type'].unique())
    
    # Sidebar Controls
    st.sidebar.markdown("### 🏷️ Filter Presets")
    st.sidebar.caption("Load saved slices into a fresh comparison table so you can keep the current view intact.")

    preset_options = ["None"] + sorted(list(st.session_state.presets.keys()))
    selected_preset = st.sidebar.selectbox("Load Preset", options=preset_options)

    if selected_preset != "None":
        st.sidebar.info(f"Preset '{selected_preset}' will create a NEW table.")
    
        if st.sidebar.button("📂 Load Preset as New Table", key="load_preset_btn", width='stretch'):
            # 1. Create new table
            st.session_state.num_tables += 1
            target_table = st.session_state.num_tables - 1
            preset_data = st.session_state.presets[selected_preset]
            # Map preset data back to session state keys for the target table
            mapping = {
                'models': (f"filter_m_{target_table}", models),
                'expr_types': (f"filter_e_{target_table}", expr_types),
                'w_bits': (f"filter_wb_{target_table}", df['w_bits'].dropna().unique()),
                'w_exp': (f"filter_we_{target_table}", df['w_exp'].dropna().unique()),
                'w_mant': (f"filter_wm_{target_table}", df['w_mant'].dropna().unique()),
                'a_bits': (f"filter_ab_{target_table}", df['a_bits'].dropna().unique()),
                'a_exp': (f"filter_ae_{target_table}", df['a_exp'].dropna().unique()),
                'a_mant': (f"filter_am_{target_table}", df['a_mant'].dropna().unique())
            }
        
            for preset_key, (session_key, options) in mapping.items():
                if preset_key in preset_data:
                    st.session_state[session_key] = normalize_filter_selection(
                        preset_data[preset_key],
                        options,
                    )
                    st.session_state[f"{session_key}_sync_required"] = True
        
            st.sidebar.success(f"Loaded '{selected_preset}'")
            st.rerun()

        if st.sidebar.button("🗑️ Delete Preset", key="delete_preset_btn"):
            del st.session_state.presets[selected_preset]
            save_presets(st.session_state.presets)
            st.sidebar.warning(f"Deleted '{selected_preset}'")
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Controls")
    if st.sidebar.button("🧹 Reset All Filters", width='stretch'):
        # Clear all filter-related keys from session state
        for key in list(st.session_state.keys()):
            if key.startswith("filter_"):
                del st.session_state[key]
        st.rerun()
    
    st.sidebar.markdown("---")

    # Deduplication Toggle
    show_newest = st.sidebar.checkbox(
        "Show only newest runs", 
        value=True, 
        help="If multiple runs exist for the same model and datatypes, only show the most recent one."
    )

    if show_newest:
        # Since get_runs() returns id DESC (newest first), keep='first' gets the newest
        df = df.drop_duplicates(
            subset=['model_name', 'experiment_type', 'weight_dt', 'activation_dt'], 
            keep='first'
        )

    # Legend for Special Values
    st.sidebar.info("""
    **Legend:**
    - **Dynamic**: Variable bit-width (e.g., dynamic input quant)
    - **Bits**: Bit-width of weights/activations
    - **Drop**: Accuracy loss relative to FP32 reference
    """)
    st.sidebar.markdown("---")
    @st.fragment(run_every=10)
    def _render_experiment_result_tables():
        with st.spinner("Refreshing experiment runs..."):
            df = get_runs(selected_run_limit)
            df = preprocess_runs_df(df)
        if df is None or df.empty:
            st.warning("No runs found in the database yet. Run an experiment first!")
            return
        df['run_date_dt'] = pd.to_datetime(df['run_date'], errors='coerce')
        if len(date_range) == 2:
            start_date, end_date = date_range
            df = df[(df['run_date_dt'].dt.date >= start_date) & (df['run_date_dt'].dt.date <= end_date)]
        df['experiment_type'] = df['experiment_type'].fillna('unknown')
        df['model_name'] = df['model_name'].fillna('unknown')
        if show_newest:
            df = df.drop_duplicates(
                subset=['model_name', 'experiment_type', 'weight_dt', 'activation_dt'],
                keep='first'
            )
        models = sorted(df['model_name'].unique())
        expr_types = sorted(df['experiment_type'].unique())
        render_dashboard_intro()
        st.markdown("---")

        # Render Tables
        for i in range(st.session_state.num_tables):
            col_header, col_rm = st.columns([8, 1])
            with col_header:
                st.markdown(
                    f"""
                    <div class="dashboard-section-title">
                        <div>
                            <h3>Comparison Table {i+1}</h3>
                            <p>Mix broad model filters with datatype constraints, then select rows to inspect or compare.</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            if st.session_state.num_tables > 1:
                if col_rm.button("🗑️", key=f"rm_table_{i}", help=f"Remove Table {i+1}"):
                    remove_table(i)

            # Local filters for each table
            st.markdown("<div class='dashboard-filter-note'>Core filters stay visible here; deeper datatype filters and preset saving live just below.</div>", unsafe_allow_html=True)
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                selected_models = render_filter_checklist(
                    f"Models (T{i+1})", 
                    options=models, 
                    session_key=f"filter_m_{i}",
                    help_text="Tick the models you want included."
                )
            with col_f2:
                selected_exprs = render_filter_checklist(
                    f"Experiment Types (T{i+1})", 
                    options=expr_types, 
                    session_key=f"filter_e_{i}",
                    help_text="Tick the experiment families you want included."
                )
    
            # Advanced DT Filtering
            with st.expander(f"Advanced Datatype Filters (T{i+1})"):
                st.caption("Use these only when you want to narrow by datatype internals such as bit-width, exponent, or mantissa.")
                weight_col, activation_col = st.columns(2)
                with weight_col:
                    st.markdown("**Weight Filters**")
                    w_bits_options = sorted(df['w_bits'].dropna().unique())
                    selected_w_bits = render_filter_checklist(
                        "Weight Bits", 
                        options=w_bits_options, 
                        session_key=f"filter_wb_{i}",
                        format_func=lambda x: "Dynamic" if x == 0 else f"{int(x)} Bits",
                    )
        
                    w_exp_options = sorted(df['w_exp'].dropna().unique())
                    selected_w_exp = render_filter_checklist(
                        "Weight Exponent", 
                        options=w_exp_options, 
                        session_key=f"filter_we_{i}",
                    )
            
                    w_mant_options = sorted(df['w_mant'].dropna().unique())
                    selected_w_mant = render_filter_checklist(
                        "Weight Mantissa", 
                        options=w_mant_options, 
                        session_key=f"filter_wm_{i}",
                    )

                with activation_col:
                    st.markdown("**Activation Filters**")
                    a_bits_options = sorted(df['a_bits'].dropna().unique())
                    selected_a_bits = render_filter_checklist(
                        "Activation Bits", 
                        options=a_bits_options, 
                        session_key=f"filter_ab_{i}",
                        format_func=lambda x: "Dynamic" if x == 0 else f"{int(x)} Bits",
                    )
            
                    a_exp_options = sorted(df['a_exp'].dropna().unique())
                    selected_a_exp = render_filter_checklist(
                        "Activation Exponent", 
                        options=a_exp_options, 
                        session_key=f"filter_ae_{i}",
                    )
            
                    a_mant_options = sorted(df['a_mant'].dropna().unique())
                    selected_a_mant = render_filter_checklist(
                        "Activation Mantissa", 
                        options=a_mant_options, 
                        session_key=f"filter_am_{i}",
                    )

                # Save Preset UI under each table's filters
                with st.expander(f"💾 Save current filters as Preset (T{i+1})"):
                    p_col1, p_col2 = st.columns([3, 1])
                    preset_name = p_col1.text_input("Preset Name", key=f"name_input_{i}", placeholder="e.g. My Optimal ResNet")
                    if p_col2.button("Save", key=f"save_btn_{i}", type="primary", width='stretch'):
                        if preset_name:
                            st.session_state.presets[preset_name] = {
                                'target_table': i,
                                'models': selected_models,
                                'expr_types': selected_exprs,
                                'w_bits': selected_w_bits,
                                'w_exp': selected_w_exp,
                                'w_mant': selected_w_mant,
                                'a_bits': selected_a_bits,
                                'a_exp': selected_a_exp,
                                'a_mant': selected_a_mant
                            }
                            save_presets(st.session_state.presets)
                            st.success(f"Saved '{preset_name}'!")
                            st.rerun()
                        else:
                            st.error("Enter a name")

            # Apply local filters
            filtered_df = df[
                (df['model_name'].isin(selected_models)) & 
                (df['experiment_type'].isin(selected_exprs))
            ].copy()

            # Helper to apply optional filters (only if user changed them)
            def apply_opt_filter(curr_df, col, selected, all_options):
                if len(selected) < len(all_options):
                    return curr_df[curr_df[col].isin(selected)]
                return curr_df

            # Bits are mandatory filters (all selected by default).
            # When all options are selected (no user narrowing), also keep rows whose
            # bits value is NaN — this happens for formats that parse_dt can't decode
            # (e.g. the hybrid "opt_layer[...]" weight_dt strings).
            w_bits_all = len(selected_w_bits) == len(w_bits_options)
            a_bits_all = len(selected_a_bits) == len(a_bits_options)
            filtered_df = filtered_df[
                filtered_df['w_bits'].isin(selected_w_bits) | (w_bits_all & filtered_df['w_bits'].isna())
            ]
            filtered_df = filtered_df[
                filtered_df['a_bits'].isin(selected_a_bits) | (a_bits_all & filtered_df['a_bits'].isna())
            ]

            # Exp/Mant are optional filters
            filtered_df = apply_opt_filter(filtered_df, 'w_exp', selected_w_exp, w_exp_options)
            filtered_df = apply_opt_filter(filtered_df, 'w_mant', selected_w_mant, w_mant_options)
            filtered_df = apply_opt_filter(filtered_df, 'a_exp', selected_a_exp, a_exp_options)
            filtered_df = apply_opt_filter(filtered_df, 'a_mant', selected_a_mant, a_mant_options)
    
            if not filtered_df.empty:
                filtered_df = _attach_effective_references(filtered_df)
                # Calculate Accuracy Drop relative to reference
                if 'ref_acc1_effective' in filtered_df.columns:
                    filtered_df['acc1_drop'] = filtered_df['ref_acc1_effective'] - filtered_df['acc1']
                    filtered_df['acc5_drop'] = filtered_df['ref_acc5_effective'] - filtered_df['acc5']
                if 'ref_certainty_effective' in filtered_df.columns and 'certainty' in filtered_df.columns:
                    filtered_df['cert_drop'] = filtered_df['ref_certainty_effective'] - filtered_df['certainty']
        
                # Clean up and reorder columns for display
                cols_to_drop = [
                    'run_date_dt', 'id', 'quant_map_json', 'input_map_json', 'config_json',
                    'ref_acc1_effective', 'ref_acc5_effective', 'ref_certainty_effective'
                ]
                display_df = filtered_df.drop(columns=[c for c in cols_to_drop if c in filtered_df.columns])
        
                # Reorder columns to put metrics and targets front and center
                main_cols = ['model_name', 'experiment_type', 'weight_dt', 'activation_dt', 'acc1', 'acc1_drop', 'acc5', 'acc5_drop', 'mse', 'l1', 'certainty', 'cert_drop', 'status', 'run_date']
                existing_main_cols = [c for c in main_cols if c in display_df.columns]
                other_cols = [c for c in display_df.columns if c not in existing_main_cols]
                display_df = display_df[existing_main_cols + other_cols]
        
                # Render Interactive Dataframe with modern Selection
                event = st.dataframe(
                    display_df, 
                    width='stretch',
                    hide_index=True,
                    selection_mode="multi-row",
                    on_select="rerun",
                    key=f"table_{i}",
                    column_config={
                        "model_name": "Model",
                        "experiment_type": "Exp. Type",
                        "weight_dt": "Weight DT",
                        "activation_dt": "Activation DT",
                        "acc1": st.column_config.NumberColumn("Acc1 (%)", format="%.2f"),
                        "acc5": st.column_config.NumberColumn("Acc5 (%)", format="%.2f"),
                        "acc1_drop": st.column_config.NumberColumn("Drop1 (%)", format="%.2f", help="Ref Acc1 - Acc1"),
                        "acc5_drop": st.column_config.NumberColumn("Drop5 (%)", format="%.2f", help="Ref Acc5 - Acc5"),
                        "mse": st.column_config.NumberColumn("MSE", format="%.2e"),
                        "l1": st.column_config.NumberColumn("L1", format="%.2e"),
                        "certainty": st.column_config.NumberColumn("Cert.", format="%.4f", help="Model prediction confidence (softmax max prob)"),
                        "cert_drop": st.column_config.NumberColumn("Cert. Drop", format="%.4f", help="Ref Certainty - Certainty"),
                        "status": "Status",
                        "run_date": "Date"
                    }
                )

                # --- Visualization Section ---
                selected_indices = event.selection.rows

                # Action buttons — appear based on what's available in selected rows.
                selected_run_rows = []
                if len(selected_indices) >= 1:
                    orig_indices = [display_df.index[j] for j in selected_indices]
                    selected_run_rows = [filtered_df.loc[idx].to_dict() for idx in orig_indices]

                selected_count = len(selected_indices)
                if selected_count:
                    st.markdown(
                        f"""
                        <div class="dashboard-selection-banner">
                            <strong>{selected_count} row{'s' if selected_count != 1 else ''} selected.</strong>
                            <span> Use the actions below to open configs, inspect layer formats, compare accuracy, rename experiments, export rows, or remove rows.</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("Select one or more rows to unlock run details, comparison charts, and deletion tools.")

                st.markdown("#### Actions")
                if len(selected_indices) == 1:
                    orig_idx = display_df.index[selected_indices[0]]
                    run_row = filtered_df.loc[orig_idx].to_dict()
                    weight_map_json, input_map_json, _ = _resolve_maps_for_display(run_row)
                    has_weight_map = (
                        _safe_json_load(weight_map_json) is not None
                    )
                    has_input_map = (
                        _safe_json_load(input_map_json) is not None
                    )
                else:
                    run_row = None
                    has_weight_map = False
                    has_input_map = False

                if len(selected_indices) >= 1:
                    rows_with_details = [
                        row for row in selected_run_rows
                        if (
                            (
                                'config_json' in row and
                                pd.notna(row.get('config_json')) and
                                row.get('config_json')
                            ) or (
                                'quant_map_json' in row and
                                pd.notna(row.get('quant_map_json')) and
                                row.get('quant_map_json')
                            ) or (
                                'input_map_json' in row and
                                pd.notna(row.get('input_map_json')) and
                                row.get('input_map_json')
                            )
                        )
                    ]
                    num_runs = len(selected_indices)
                    if rows_with_details:
                        label = (
                            f"⚙️ View Config + Win Rates ({len(rows_with_details)})"
                            if len(rows_with_details) > 1 else "⚙️ View Config + Win Rates"
                        )

                    selected_run_ids = []
                    for row in selected_run_rows:
                        run_id = row.get('id')
                        if pd.notna(run_id):
                            selected_run_ids.append(int(run_id))
                    selected_run_ids = sorted(set(selected_run_ids))

                    if selected_count:
                        st.caption(f"Ready to work with {num_runs} selected run{'s' if num_runs != 1 else ''}.")

                    action_specs = []
                    if len(selected_indices) == 1 and (has_weight_map or has_input_map):
                        action_specs.append({
                            "label": "🔬 View Layer Formats",
                            "key": f"btn_layers_{i}",
                            "type": "secondary",
                            "handler": lambda run_row=run_row: show_layer_breakdown(run_row),
                        })
                    if rows_with_details:
                        action_specs.append({
                            "label": label,
                            "key": f"btn_config_{i}",
                            "type": "secondary",
                            "handler": lambda rows_with_details=rows_with_details: show_run_config(rows_with_details),
                        })
                    action_specs.append({
                        "label": f"📈 Generate Comparison ({num_runs})",
                        "key": f"btn_plot_{i}",
                        "type": "primary",
                        "handler": None,
                    })
                    if selected_run_ids:
                        action_specs.append({
                            "label": f"✏️ Rename Experiment ({len(selected_run_ids)})",
                            "key": f"btn_edit_experiment_{i}",
                            "type": "secondary",
                            "handler": lambda selected_run_rows=selected_run_rows, table_index=i: show_edit_experiment_dialog(selected_run_rows, table_index),
                        })
                        action_specs.append({
                            "label": f"💾 Create DB ({len(selected_run_ids)})",
                            "key": f"btn_create_db_{i}",
                            "type": "secondary",
                            "handler": lambda selected_run_rows=selected_run_rows, table_index=i: show_create_db_dialog(selected_run_rows, table_index),
                        })
                    if selected_run_ids:
                        delete_label = (
                            f"🗑️ Delete Selected Rows ({len(selected_run_ids)})"
                            if len(selected_run_ids) > 1 else "🗑️ Delete Selected Row"
                        )
                        action_specs.append({
                            "label": delete_label,
                            "key": f"btn_delete_rows_{i}",
                            "type": "secondary",
                            "handler": lambda selected_run_rows=selected_run_rows, table_index=i: show_delete_runs_dialog(selected_run_rows, table_index),
                        })

                    action_cols = st.columns(len(action_specs))
                    run_comparison = False
                    for col, spec in zip(action_cols, action_specs):
                        if col.button(
                            spec["label"],
                            key=spec["key"],
                            type=spec["type"],
                            width='stretch',
                        ):
                            if spec["key"] == f"btn_plot_{i}":
                                run_comparison = True
                            elif spec["handler"] is not None:
                                spec["handler"]()

                    if run_comparison:
                        with st.spinner(f"Building comparison chart for {num_runs} runs..."):
                            orig_indices = [display_df.index[j] for j in selected_indices]
                            selected_df = filtered_df.loc[orig_indices].copy()

                            # Create labels
                            selected_df['run_label'] = (
                                selected_df['model_name'].astype(str) + "\n" +
                                selected_df['experiment_type'].astype(str) + "\n" +
                                selected_df['weight_dt'].astype(str) + "/" +
                                selected_df['activation_dt'].astype(str)
                            )

                            # Prepare flattened data
                            chart_data = []

                            # Group by model to show ref once per model
                            models_in_selection = selected_df['model_name'].unique()

                            for model in models_in_selection:
                                model_rows = selected_df[selected_df['model_name'] == model]

                                # 1. Add Reference Entry (ONCE)
                                first_row = model_rows.iloc[0]
                                ref_label = f"{model}\nREF\nfp32/fp32"
                                chart_data.append({'Label': ref_label, 'MetricName': 'Acc1', 'MetricType': 'Reference (Acc1)', 'Accuracy (%)': first_row['ref_acc1_effective'] if 'ref_acc1_effective' in first_row else 0})
                                chart_data.append({'Label': ref_label, 'MetricName': 'Acc5', 'MetricType': 'Reference (Acc5)', 'Accuracy (%)': first_row['ref_acc5_effective'] if 'ref_acc5_effective' in first_row else 0})

                                # 2. Add Quantized Entries
                                for _, row in model_rows.iterrows():
                                    quant_label = (
                                        f"{model}\n"
                                        f"{row.get('experiment_type', '')}\n"
                                        f"{row['weight_dt']}/{row['activation_dt']}"
                                    )
                                    chart_data.append({'Label': quant_label, 'MetricName': 'Acc1', 'MetricType': 'Quantized (Acc1)', 'Accuracy (%)': row['acc1']})
                                    chart_data.append({'Label': quant_label, 'MetricName': 'Acc5', 'MetricType': 'Quantized (Acc5)', 'Accuracy (%)': row['acc5']})

                            chart_df = pd.DataFrame(chart_data)
                        show_large_chart(chart_df, selected_df)
                else:
                    st.caption("Pick rows above to enable the action buttons.")
            else:
                st.warning("No rows match the current filter set. Relax a model, experiment type, or datatype filter to repopulate the table.")
    
            st.markdown("---")
    
            # # --- Model Architecture Graph Visualization ---
            # st.markdown(f"#### 🏗️ Model Architecture & Quantization")
            # if not filtered_df.empty:
            #     available_models = sorted(filtered_df['model_name'].unique())
            #     graph_status_col, graph_action_col = st.columns([3, 2])
            #     graph_status_col.caption(
            #         "Open the dedicated graph viewer to generate a fresh architecture graph from the current code."
            #     )
            #     if graph_action_col.button("🏗️ Open Graph Viewer", key=f"open_graph_viewer_{i}", type="primary", width='stretch'):
            #         show_graph_viewer(i, available_models)
            # else:
            #     st.info("No data to display. Apply filters above to see models.")
    
            # st.markdown("---")

        # Add Table Button
        st.button("➕ Add Table", on_click=add_table)

    _render_experiment_result_tables()


# Close the Experiments tab context.
tab_exp.__exit__(None, None, None)
