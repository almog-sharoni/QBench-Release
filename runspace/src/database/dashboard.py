import streamlit as st
import pandas as pd
import os
import sys

# Add project root to sys.path
# runspace/src/database/dashboard.py -> runspace/src/database -> runspace/src -> runspace -> root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.src.database.handler import RunDatabase
import json
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

PRESETS_FILE = os.path.join(os.path.dirname(__file__), "presets.json")
DB_PATH = os.path.join(PROJECT_ROOT, "runspace/database/runs.db")
RUN_WINDOW_TO_LIMIT = {
    "200 (fastest)": 200,
    "500": 500,
    "1000": 1000,
    "All": None,
}

def load_presets():
    if os.path.exists(PRESETS_FILE):
        try:
            with open(PRESETS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_presets(presets):
    with open(PRESETS_FILE, 'w') as f:
        json.dump(presets, f, indent=4, cls=NpEncoder)


def inject_global_styles():
    st.markdown("""
    <style>
    .stButton > button {
        transition: background-color 0.1s ease, border 0.1s ease !important;
        border-radius: 4px !important;
    }

    div[data-testid="stHorizontalBlock"] .stButton > button {
        background-color: #fb7185 !important;
        color: white !important;
        border: 1px solid #e11d48 !important;
        padding: 0px 8px !important;
        min-height: 24px !important;
        height: 24px !important;
        font-size: 10px !important;
        line-height: normal !important;
        font-weight: 600;
        white-space: nowrap !important;
        width: 100% !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }

    div[data-testid="stHorizontalBlock"] .stButton > button[kind="primary"],
    div[data-testid="stHorizontalBlock"] .stButton > button[data-testid*="primary"] {
        background-color: #10b981 !important;
        color: white !important;
        border: 1px solid #059669 !important;
    }

    div[data-testid="stHorizontalBlock"] {
        gap: 6px !important;
    }

    [data-testid="stDialog"] [data-testid="stVerticalBlock"] {
        overflow-x: auto !important;
        padding-bottom: 20px;
    }

    .element-container:has(iframe) {
        min-width: fit-content;
    }
    </style>
    """, unsafe_allow_html=True)

def parse_dt(dt_str):
    """Extract (bits, exp, mant) from DT strings like 'fp4_e1m2'."""
    if not dt_str or not isinstance(dt_str, str):
        return None, None, None
    dt_clean = dt_str.lower().strip()
    if dt_clean == 'fp32': return 32, None, None
    if dt_clean == 'fp16': return 16, None, None
    if dt_clean == 'bf16': return 16, None, None
    
    bits, exp, mant = None, None, None
    parts = dt_clean.split('_')
    if parts[0].startswith('fp'):
        try: bits = int(parts[0][2:])
        except: pass
    elif parts[0] == 'dyn':
        bits = 0 # Sentinel for Dynamic
    if len(parts) > 1:
        em = parts[1] # e1m2 or e1
        if 'e' in em:
            try:
                if 'm' in em: # e1m2
                    exp = int(em.split('m')[0][1:])
                    mant = int(em.split('m')[1])
                else: # e1
                    exp = int(em[1:])
            except: pass
    return bits, exp, mant


def get_runs(limit):
    db = RunDatabase(db_path=DB_PATH)
    return db.get_runs(limit=limit)


def delete_runs_by_ids(run_ids):
    db = RunDatabase(db_path=DB_PATH)
    return db.delete_runs_by_ids(run_ids)


def preprocess_runs_df(df):
    if df is None or df.empty:
        return df

    parsed_df = df.copy()
    for col in ['weight_dt', 'activation_dt']:
        prefix = 'w' if col.startswith('weight') else 'a'
        parsed = parsed_df[col].apply(parse_dt)
        parsed_df[f'{prefix}_bits'] = parsed.apply(lambda x: x[0])
        parsed_df[f'{prefix}_exp'] = parsed.apply(lambda x: x[1])
        parsed_df[f'{prefix}_mant'] = parsed.apply(lambda x: x[2])
    return parsed_df


def _attach_effective_references(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure each row has usable reference metrics even when legacy rows logged
    ref_* as zeros. Prefers latest fp32_ref per model, falls back to latest
    fp32/fp32 row with positive accuracy.
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    for col in ('acc1', 'acc5', 'ref_acc1', 'ref_acc5', 'certainty', 'ref_certainty'):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce')

    refs = out.copy()
    if 'weight_dt' in refs.columns and 'activation_dt' in refs.columns:
        refs = refs[
            refs['weight_dt'].astype(str).str.lower().eq('fp32') &
            refs['activation_dt'].astype(str).str.lower().eq('fp32')
        ]
    if 'acc1' in refs.columns:
        refs = refs[refs['acc1'].fillna(0) > 0]

    if refs.empty:
        out['ref_acc1_effective'] = out.get('ref_acc1', 0.0).fillna(0.0)
        out['ref_acc5_effective'] = out.get('ref_acc5', 0.0).fillna(0.0)
        out['ref_certainty_effective'] = out.get('ref_certainty', 0.0).fillna(0.0)
        return out

    refs = refs.copy()
    refs['is_fp32_ref'] = refs.get('experiment_type', '').astype(str).eq('fp32_ref').astype(int)
    refs['is_success'] = refs.get('status', '').astype(str).eq('SUCCESS').astype(int)
    sort_cols = [c for c in ['model_name', 'is_fp32_ref', 'is_success', 'run_date', 'id'] if c in refs.columns]
    sort_asc = [True, False, False, False, False][:len(sort_cols)]
    refs = refs.sort_values(by=sort_cols, ascending=sort_asc)
    refs = refs.drop_duplicates(subset=['model_name'], keep='first')

    ref_acc1_map = refs.set_index('model_name')['acc1'].to_dict() if 'acc1' in refs.columns else {}
    ref_acc5_map = refs.set_index('model_name')['acc5'].to_dict() if 'acc5' in refs.columns else {}
    ref_cert_map = refs.set_index('model_name')['certainty'].to_dict() if 'certainty' in refs.columns else {}

    out['ref_acc1_effective'] = out.get('ref_acc1', 0.0)
    out['ref_acc5_effective'] = out.get('ref_acc5', 0.0)
    out['ref_certainty_effective'] = out.get('ref_certainty', 0.0)

    if 'model_name' in out.columns:
        model_ref_acc1 = out['model_name'].map(ref_acc1_map)
        model_ref_acc5 = out['model_name'].map(ref_acc5_map)
        model_ref_cert = out['model_name'].map(ref_cert_map)
    else:
        model_ref_acc1 = pd.Series([np.nan] * len(out), index=out.index)
        model_ref_acc5 = pd.Series([np.nan] * len(out), index=out.index)
        model_ref_cert = pd.Series([np.nan] * len(out), index=out.index)

    miss_ref1 = out['ref_acc1_effective'].isna() | (out['ref_acc1_effective'] <= 0)
    miss_ref5 = out['ref_acc5_effective'].isna() | (out['ref_acc5_effective'] <= 0)
    miss_refc = out['ref_certainty_effective'].isna() | (out['ref_certainty_effective'] <= 0)
    out.loc[miss_ref1, 'ref_acc1_effective'] = model_ref_acc1[miss_ref1]
    out.loc[miss_ref5, 'ref_acc5_effective'] = model_ref_acc5[miss_ref5]
    out.loc[miss_refc, 'ref_certainty_effective'] = model_ref_cert[miss_refc]

    # fp32_ref rows should use themselves as reference.
    if 'experiment_type' in out.columns and 'acc1' in out.columns and 'acc5' in out.columns:
        is_ref = out['experiment_type'].astype(str).eq('fp32_ref')
        out.loc[is_ref & out['acc1'].notna(), 'ref_acc1_effective'] = out.loc[is_ref & out['acc1'].notna(), 'acc1']
        out.loc[is_ref & out['acc5'].notna(), 'ref_acc5_effective'] = out.loc[is_ref & out['acc5'].notna(), 'acc5']
        if 'certainty' in out.columns:
            out.loc[is_ref & out['certainty'].notna(), 'ref_certainty_effective'] = out.loc[is_ref & out['certainty'].notna(), 'certainty']

    out['ref_acc1_effective'] = out['ref_acc1_effective'].fillna(0.0)
    out['ref_acc5_effective'] = out['ref_acc5_effective'].fillna(0.0)
    out['ref_certainty_effective'] = out['ref_certainty_effective'].fillna(0.0)
    return out


def _get_format_bits(fmt):
    """Best-effort bit width extraction for strings like fp6_e2m3."""
    if not fmt:
        return 32
    text = str(fmt).strip().lower()
    if text == "fp32":
        return 32
    if text == "fp16" or text == "bf16":
        return 16
    if text == "int8":
        return 8
    if text == "int4":
        return 4
    if text.startswith("fp"):
        base = text.split("_", 1)[0]
        try:
            return int(base.replace("fp", ""))
        except Exception:
            return 32
    return 32


def _sort_quant_formats(formats):
    """Sort by bit width desc then exponent bits desc, similar to plotting utils."""
    def parse_fmt(fmt):
        text = str(fmt).strip().lower()
        bits = _get_format_bits(text)
        exp = 0
        if "_e" in text:
            try:
                exp_part = text.split("_e", 1)[1]
                exp = int(exp_part.split("m", 1)[0])
            except Exception:
                exp = 0
        return bits, exp, text

    return sorted(set(formats), key=parse_fmt, reverse=True)


def _safe_json_load(raw_json):
    if raw_json is None:
        return None
    if isinstance(raw_json, float) and pd.isna(raw_json):
        return None
    if isinstance(raw_json, (dict, list)):
        return raw_json
    try:
        return json.loads(raw_json)
    except Exception:
        return None


def _compute_weight_win_rate_views(raw_json):
    """
    Build summary tables for layer/chunk winners from quant_map_json.
    Returns (summary_df, layer_df, layer_chunk_df, meta) or (None, None, None, None) if unavailable.
    """
    quant_map = _safe_json_load(raw_json)
    if not isinstance(quant_map, dict) or not quant_map:
        return None, None, None, None

    layer_rows = []
    layer_chunk_rows = []
    layer_win_counts = {}
    chunk_win_counts = {}

    for layer_idx, (layer, value) in enumerate(quant_map.items()):
        layer_type = "?"
        fmt_spec = value
        explicit_counts = None
        explicit_total_chunks = None
        dominant_format = None

        if isinstance(value, dict):
            layer_type = str(value.get("type", "?"))
            fmt_spec = value.get("format")
            if isinstance(value.get("format_counts"), dict):
                explicit_counts = {}
                for fmt, cnt in value["format_counts"].items():
                    try:
                        explicit_counts[str(fmt)] = int(cnt)
                    except Exception:
                        continue
            try:
                if value.get("total_chunks") is not None:
                    explicit_total_chunks = int(value.get("total_chunks"))
            except Exception:
                explicit_total_chunks = None
            if value.get("dominant_format") is not None:
                dominant_format = str(value.get("dominant_format"))

        counts = {}
        if explicit_counts:
            counts = explicit_counts
        elif isinstance(fmt_spec, list):
            for fmt in fmt_spec:
                key = str(fmt)
                counts[key] = counts.get(key, 0) + 1
        elif fmt_spec is not None:
            key = str(fmt_spec)
            counts[key] = counts.get(key, 0) + 1

        if not counts:
            continue

        if dominant_format is None:
            dominant_format = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
        total_chunks = explicit_total_chunks if explicit_total_chunks is not None else int(sum(counts.values()))
        if total_chunks <= 0:
            total_chunks = int(sum(counts.values()))

        layer_win_counts[dominant_format] = layer_win_counts.get(dominant_format, 0) + 1
        for fmt, cnt in counts.items():
            chunk_win_counts[fmt] = chunk_win_counts.get(fmt, 0) + int(cnt)
            layer_chunk_rows.append({
                "Layer": layer,
                "Layer Index": int(layer_idx),
                "Type": layer_type,
                "Format": fmt,
                "Chunk Wins": int(cnt),
            })

        layer_rows.append({
            "Layer": layer,
            "Layer Index": int(layer_idx),
            "Type": layer_type,
            "Dominant Format": dominant_format,
            "Chunks": int(total_chunks),
        })

    if not layer_rows:
        return None, None, None, None

    layer_total = len(layer_rows)
    chunk_total = int(sum(chunk_win_counts.values()))
    all_formats = _sort_quant_formats(set(layer_win_counts.keys()) | set(chunk_win_counts.keys()))

    summary_rows = []
    for fmt in all_formats:
        layer_wins = int(layer_win_counts.get(fmt, 0))
        chunk_wins = int(chunk_win_counts.get(fmt, 0))
        summary_rows.append({
            "Format": fmt,
            "Layer Wins": layer_wins,
            "Layer Win Rate (%)": (100.0 * layer_wins / layer_total) if layer_total > 0 else 0.0,
            "Chunk Wins": chunk_wins,
            "Chunk Win Rate (%)": (100.0 * chunk_wins / chunk_total) if chunk_total > 0 else 0.0,
        })

    summary_df = pd.DataFrame(summary_rows)
    layer_df = pd.DataFrame(layer_rows).sort_values(by=["Layer Index", "Layer"], ascending=[True, True])
    layer_chunk_df = pd.DataFrame(layer_chunk_rows)
    layer_chunk_df = layer_chunk_df.merge(
        layer_df[["Layer", "Layer Index", "Chunks"]],
        on=["Layer", "Layer Index"],
        how="left"
    ).sort_values(
        by=["Layer Index", "Layer", "Format"],
        ascending=[True, True, True]
    )

    top_layer_format = max(layer_win_counts.items(), key=lambda x: (x[1], x[0]))[0] if layer_win_counts else "-"
    top_chunk_format = max(chunk_win_counts.items(), key=lambda x: (x[1], x[0]))[0] if chunk_win_counts else "-"
    meta = {
        "layers": layer_total,
        "chunks": chunk_total,
        "top_layer_format": top_layer_format,
        "top_chunk_format": top_chunk_format,
    }
    return summary_df, layer_df, layer_chunk_df, meta


def get_model_graph_json(model_name):
    db = RunDatabase(db_path=DB_PATH)
    return db.get_model_graph_json(model_name)


def get_model_graph_metadata(model_name):
    db = RunDatabase(db_path=DB_PATH)
    return db.get_model_graph_metadata(model_name)


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
            use_container_width=True,
        )

    st.dataframe(
        summary_df,
        use_container_width=True,
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
            use_container_width=True,
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

st.set_page_config(page_title="QBench Experiment Dashboard", layout="wide")

st.title("🚀 QBench Experiment Tracker")
inject_global_styles()
flash_message = st.session_state.pop("dashboard_flash_message", None)
if flash_message:
    st.success(flash_message)

# Initial Session State for Presets
if 'presets' not in st.session_state:
    st.session_state.presets = load_presets()

st.sidebar.header("⚡ Performance")
run_window_label = st.sidebar.selectbox(
    "Rows to Load",
    options=list(RUN_WINDOW_TO_LIMIT.keys()),
    index=1,
    help="Loads only the newest N runs to keep reruns fast. Use 'All' when you need full history.",
)
selected_run_limit = RUN_WINDOW_TO_LIMIT[run_window_label]
load_graphs_on_demand = st.sidebar.checkbox(
    "Load Graphs On Demand",
    value=True,
    help="Delays model graph JSON fetch/decompression until you explicitly request a graph.",
)
st.sidebar.caption("Use `Refresh Data` after new experiments complete.")

st.markdown("---")

with st.spinner("Loading experiment runs..."):
    df = get_runs(selected_run_limit)

if df.empty:
    st.warning("No runs found in the database yet. Run an experiment first!")
else:
    with st.spinner("Preparing dashboard data..."):
        df = preprocess_runs_df(df)
    st.caption(f"Showing {len(df)} runs from the selected load window ({run_window_label}).")
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
                else:
                    fmt = value
                    layer_type = "?"
                    per_chunk_counts = None
                    total_chunks = None

                if isinstance(fmt, list):
                    counts = {}
                    for f in fmt:
                        counts[f] = counts.get(f, 0) + 1
                    fmt_str = ", ".join(
                        f"{f}×{c}" for f, c in sorted(counts.items(), key=lambda x: -x[1])
                    )
                    rows.append({"Layer": layer, "Type": layer_type, "Format": fmt_str, "Mode": "per-chunk"})
                else:
                    rows.append({"Layer": layer, "Type": layer_type, "Format": str(fmt), "Mode": mode_label})

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
                use_container_width=True,
                column_config={
                    "Layer":  st.column_config.TextColumn("Layer", width="large"),
                    "Type":   st.column_config.TextColumn("Type"),
                    "Format": st.column_config.TextColumn("Format"),
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
                    st.dataframe(chunk_df, use_container_width=True)

        if not is_input_exp:
            _render_map_section("⚖️ Weight Formats", weight_map_json, "per-layer")
        _render_map_section("⚡ Input Formats",  input_map_json,  "dynamic")

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

        st.warning("This permanently deletes the selected rows from `runs.db`. This action cannot be undone.")

        preview_cols = [
            'id', 'model_name', 'experiment_type', 'weight_dt', 'activation_dt',
            'acc1', 'acc5', 'status', 'run_date'
        ]
        existing_preview_cols = [c for c in preview_cols if c in selected_df.columns]
        st.dataframe(
            selected_df[existing_preview_cols],
            use_container_width=True,
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
            use_container_width=True,
        ):
            deleted_count = delete_runs_by_ids(run_ids)
            st.session_state["dashboard_flash_message"] = (
                f"Deleted {deleted_count} run{'s' if deleted_count != 1 else ''} from the database."
            )
            for key in (
                f"table_{table_index}",
                f"graph_payload_{table_index}",
                f"graph_meta_{table_index}",
                f"graph_payload_model_{table_index}",
            ):
                st.session_state.pop(key, None)
            st.rerun()

    @st.dialog("📈 Accuracy Comparison", width="large")
    def show_large_chart(chart_df):
        num_groups = chart_df['Label'].nunique()
        chart_width = max(1100, num_groups * 160)
        
        st.vega_lite_chart(chart_df, {
            'mark': {'type': 'bar', 'tooltip': True, 'size': 35},
            'config': {'view': {'stroke': 'transparent'}},
            'encoding': {
                'x': {
                    'field': 'Label', 
                    'type': 'nominal', 
                    'axis': {
                        'labelAngle': -45, 
                        'title': '', 
                        'labelFontSize': 10,
                        'labelOverlap': False,
                        'labelLimit': 0
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
        }, use_container_width=False)
        st.info("💡 Tip: Use the horizontal scrollbar above to see all models. Click '...' to save.")
    # --- Multi-Table Session State ---
    if 'num_tables' not in st.session_state:
        st.session_state.num_tables = 1

    def toggle_button_group(label, options, session_key, default_state=True, format_func=str):
        st.markdown(f"**{label}**")
        
        # Initialize state explicitly if missing
        if session_key not in st.session_state:
            st.session_state[session_key] = {opt: True for opt in options}
        state = st.session_state[session_key]
        
        # Ensure all current options are mapped in the state
        for opt in options:
            if opt not in state:
                state[opt] = True
        
        # Global selection buttons
        c1, c2, _ = st.columns([0.75, 0.75, 3])
        if c1.button("Turn All On", key=f"{session_key}_all_on", type="primary", use_container_width=True):
            for opt in options:
                state[opt] = True
            st.rerun()
        if c2.button("Turn All Off", key=f"{session_key}_all_off", type="secondary", use_container_width=True):
            for opt in options:
                state[opt] = False
            st.rerun()
            
        # Draw dynamic adaptive columns for toggle buttons
        if not options: return []
        max_label_len = max(len(format_func(o)) for o in options)
        num_cols = 2 if max_label_len > 15 else (3 if max_label_len > 10 else (4 if max_label_len > 6 else 7))
        
        cols = st.columns(num_cols)
        for idx, opt in enumerate(options):
            is_on = state[opt]
            btn_label = format_func(opt)
            
            with cols[idx % num_cols]:
                if st.button(btn_label, key=f"{session_key}_btn_{idx}", type="primary" if is_on else "secondary", use_container_width=True):
                    state[opt] = not is_on
                    st.rerun()
                    
        # Return list of selected strings natively compatible with downstream logic
        return [opt for opt, is_on in state.items() if is_on]
        
    def add_table():
        st.session_state.num_tables += 1

    def remove_table(index):
        if st.session_state.num_tables > 1:
            # Shift session state for all tables above the removed index
            for i in range(index, st.session_state.num_tables - 1):
                keys_to_shift = ['filter_m', 'filter_e', 'filter_wb', 'filter_we', 'filter_wm', 'filter_ab', 'filter_ae', 'filter_am', 'name_input']
                for prefix in keys_to_shift:
                    old_key = f"{prefix}_{i+1}"
                    new_key = f"{prefix}_{i}"
                    if old_key in st.session_state:
                        st.session_state[new_key] = st.session_state[old_key]
            
            # Delete the last table's keys
            last_idx = st.session_state.num_tables - 1
            for prefix in ['filter_m', 'filter_e', 'filter_wb', 'filter_we', 'filter_wm', 'filter_ab', 'filter_ae', 'filter_am', 'name_input']:
                if f"{prefix}_{last_idx}" in st.session_state:
                    del st.session_state[f"{prefix}_{last_idx}"]
                    
            st.session_state.num_tables -= 1
            st.rerun()

    # --- Global Filters & Settings ---
    st.sidebar.header("Global Filters & Settings")
    
    # Date Filter
    df['run_date_dt'] = pd.to_datetime(df['run_date'])
    min_date = df['run_date_dt'].min().date()
    max_date = df['run_date_dt'].max().date()
    date_range = st.sidebar.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

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
    
    preset_options = ["None"] + sorted(list(st.session_state.presets.keys()))
    selected_preset = st.sidebar.selectbox("Load Preset", options=preset_options)
    
    if selected_preset != "None":
        st.sidebar.info(f"Preset '{selected_preset}' will create a NEW table.")
        
        if st.sidebar.button("📂 Load Preset as New Table", key="load_preset_btn", use_container_width=True):
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
                    val = preset_data[preset_key]
                    if isinstance(val, list):
                        # Validate: Only keep items that actually exist in current options
                        # Convert legacy list format to dictionary toggle schema mappings
                        valid_val = [v for v in val if v in options]
                        st.session_state[session_key] = {o: (o in valid_val) for o in options}
                    elif isinstance(val, dict):
                        st.session_state[session_key] = val
                    else:
                        st.session_state[session_key] = val
            
            st.sidebar.success(f"Loaded '{selected_preset}'")
            st.rerun()

        if st.sidebar.button("🗑️ Delete Preset", key="delete_preset_btn"):
            del st.session_state.presets[selected_preset]
            save_presets(st.session_state.presets)
            st.sidebar.warning(f"Deleted '{selected_preset}'")
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Controls")
    if st.sidebar.button("🔄 Refresh Data", type="primary", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key.startswith("graph_payload_") or key.startswith("graph_meta_"):
                del st.session_state[key]
        st.rerun()
    
    if st.sidebar.button("🧹 Reset All Filters", use_container_width=True):
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

    # Render Tables
    for i in range(st.session_state.num_tables):
        col_header, col_rm = st.columns([8, 1])
        col_header.subheader(f"Table {i+1}")
        if st.session_state.num_tables > 1:
            if col_rm.button("🗑️", key=f"rm_table_{i}", help=f"Remove Table {i+1}"):
                remove_table(i)
        
        # Local filters for each table
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            selected_models = toggle_button_group(
                f"Models (T{i+1})", 
                options=models, 
                session_key=f"filter_m_{i}",
                default_state=True
            )
        with col_f2:
            selected_exprs = toggle_button_group(
                f"Experiment Types (T{i+1})", 
                options=expr_types, 
                session_key=f"filter_e_{i}",
                default_state=True
            )
        
        # Advanced DT Filtering
        with st.expander(f"Advanced Datatype Filters (T{i+1})"):
            col_bits, _ = st.columns([1,1])   # left column narrower
            with col_bits:
                w_bits_options = sorted(df['w_bits'].dropna().unique())
                selected_w_bits = toggle_button_group(
                    "Weight Bits", 
                    options=w_bits_options, 
                    session_key=f"filter_wb_{i}",
                    default_state=True,
                    format_func=lambda x: "Dynamic" if x == 0 else f"{int(x)} Bits"
                )
            
                w_exp_options = sorted(df['w_exp'].dropna().unique())
                selected_w_exp = toggle_button_group(
                    "Weight Exponent", 
                    options=w_exp_options, 
                    session_key=f"filter_we_{i}",
                    default_state=True
                )
                
                w_mant_options = sorted(df['w_mant'].dropna().unique())
                selected_w_mant = toggle_button_group(
                    "Weight Mantissa", 
                    options=w_mant_options, 
                    session_key=f"filter_wm_{i}",
                    default_state=True
                )

                a_bits_options = sorted(df['a_bits'].dropna().unique())
                selected_a_bits = toggle_button_group(
                    "Activation Bits", 
                    options=a_bits_options, 
                    session_key=f"filter_ab_{i}",
                    default_state=True,
                    format_func=lambda x: "Dynamic" if x == 0 else f"{int(x)} Bits"
                )
                
                a_exp_options = sorted(df['a_exp'].dropna().unique())
                selected_a_exp = toggle_button_group(
                    "Activation Exponent", 
                    options=a_exp_options, 
                    session_key=f"filter_ae_{i}",
                    default_state=True
                )
                
                a_mant_options = sorted(df['a_mant'].dropna().unique())
                selected_a_mant = toggle_button_group(
                    "Activation Mantissa", 
                    options=a_mant_options, 
                    session_key=f"filter_am_{i}",
                    default_state=True
                )

            # Save Preset UI under each table's filters
            with st.expander(f"💾 Save current filters as Preset (T{i+1})"):
                p_col1, p_col2 = st.columns([3, 1])
                preset_name = p_col1.text_input("Preset Name", key=f"name_input_{i}", placeholder="e.g. My Optimal ResNet")
                if p_col2.button("Save", key=f"save_btn_{i}", type="primary", use_container_width=True):
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
                if has_weight_map or has_input_map:
                    if st.button("🔬 View Layer Formats", key=f"btn_layers_{i}"):
                        show_layer_breakdown(run_row)

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
                if rows_with_details:
                    label = (
                        f"⚙️ View Config + Win Rates ({len(rows_with_details)})"
                        if len(rows_with_details) > 1 else "⚙️ View Config + Win Rates"
                    )
                    if st.button(label, key=f"btn_config_{i}"):
                        show_run_config(rows_with_details)

                selected_run_ids = []
                for row in selected_run_rows:
                    run_id = row.get('id')
                    if pd.notna(run_id):
                        selected_run_ids.append(int(run_id))
                selected_run_ids = sorted(set(selected_run_ids))
                if selected_run_ids:
                    delete_label = (
                        f"🗑️ Delete Selected Rows ({len(selected_run_ids)})"
                        if len(selected_run_ids) > 1 else "🗑️ Delete Selected Row"
                    )
                    if st.button(delete_label, key=f"btn_delete_rows_{i}", use_container_width=True):
                        show_delete_runs_dialog(selected_run_rows, i)

            st.markdown(f"#### 📊 Visualization Options")
            # viz_all = st.checkbox("Visualize ALL filtered runs (ignores table selection)", key=f"viz_all_{i}")
            viz_all = False
            if viz_all or selected_indices:
                num_runs = len(display_df) if viz_all else len(selected_indices)
                st.success(f"Ready to visualize {num_runs} runs.")
                
                if st.button(f"📈 Generate Comparison ({num_runs})", key=f"btn_plot_{i}", type="primary", use_container_width=True):
                    with st.spinner(f"Building comparison chart for {num_runs} runs..."):
                        if viz_all:
                            selected_df = display_df.copy()
                        else:
                            selected_df = display_df.iloc[selected_indices].copy()

                        # Create labels
                        selected_df['run_label'] = selected_df['model_name'] + " (" + \
                                                  selected_df['weight_dt'].astype(str) + "/" + \
                                                  selected_df['activation_dt'].astype(str) + ")"

                        # Prepare flattened data
                        chart_data = []

                        # Group by model to show ref once per model
                        models_in_selection = selected_df['model_name'].unique()

                        for model in models_in_selection:
                            model_rows = selected_df[selected_df['model_name'] == model]

                            # 1. Add Reference Entry (ONCE)
                            first_row = model_rows.iloc[0]
                            ref_label = f"REF: {model}"
                            chart_data.append({'Label': ref_label, 'MetricName': 'Acc1', 'MetricType': 'Reference (Acc1)', 'Accuracy (%)': first_row['ref_acc1_effective'] if 'ref_acc1_effective' in first_row else 0})
                            chart_data.append({'Label': ref_label, 'MetricName': 'Acc5', 'MetricType': 'Reference (Acc5)', 'Accuracy (%)': first_row['ref_acc5_effective'] if 'ref_acc5_effective' in first_row else 0})

                            # 2. Add Quantized Entries
                            for _, row in model_rows.iterrows():
                                quant_label = f"{row['weight_dt']}/{row['activation_dt']} ({model})"
                                chart_data.append({'Label': quant_label, 'MetricName': 'Acc1', 'MetricType': 'Quantized (Acc1)', 'Accuracy (%)': row['acc1']})
                                chart_data.append({'Label': quant_label, 'MetricName': 'Acc5', 'MetricType': 'Quantized (Acc5)', 'Accuracy (%)': row['acc5']})

                        chart_df = pd.DataFrame(chart_data)
                    show_large_chart(chart_df)
            else:
                st.info("👆 Select rows in the table above to generate a chart.")
        else:
            st.info("No data for selected filters.")
        
        st.markdown("---")
        
        # --- Model Architecture Graph Visualization ---
        st.markdown(f"#### 🏗️ Model Architecture & Quantization")

        graph_enabled_key = f"graph_section_enabled_{i}"
        graph_payload_key = f"graph_payload_{i}"
        graph_meta_key = f"graph_meta_{i}"
        graph_model_loaded_key = f"graph_payload_model_{i}"
        if graph_enabled_key not in st.session_state:
            st.session_state[graph_enabled_key] = not load_graphs_on_demand

        if not filtered_df.empty:
            if load_graphs_on_demand and not st.session_state.get(graph_enabled_key, False):
                if st.button("⚡ Enable Architecture Graph Tools", key=f"enable_graph_section_{i}", use_container_width=True):
                    st.session_state[graph_enabled_key] = True
                    st.rerun()
                st.info("Graph data is fetched only when needed. Enable the tools to start loading graph assets.")
                st.markdown("---")
                continue

            # Get unique models in filtered data
            available_models = sorted(filtered_df['model_name'].unique())
            selected_model = st.selectbox(
                f"View quantization graph for (T{i+1})",
                options=available_models,
                key=f"graph_model_{i}"
            )
            
            if selected_model:
                loaded_model = st.session_state.get(graph_model_loaded_key)
                should_load_now = False
                if load_graphs_on_demand:
                    lcol, rcol = st.columns([2, 1])
                    if lcol.button("📊 Load Selected Graph", key=f"load_graph_{i}", type="primary", use_container_width=True):
                        should_load_now = True
                    if rcol.button("🧹 Unload Graph", key=f"unload_graph_{i}", use_container_width=True):
                        for key in (graph_payload_key, graph_meta_key, graph_model_loaded_key):
                            st.session_state.pop(key, None)
                        st.rerun()
                else:
                    should_load_now = (
                        loaded_model != selected_model or
                        graph_payload_key not in st.session_state
                    )

                if should_load_now:
                    with st.spinner(f"Loading architecture graph for {selected_model}..."):
                        graph_json, _ = get_model_graph_json(selected_model)
                        graph_meta = get_model_graph_metadata(selected_model)
                    st.session_state[graph_payload_key] = graph_json
                    st.session_state[graph_meta_key] = graph_meta
                    st.session_state[graph_model_loaded_key] = selected_model

                graph_json = st.session_state.get(graph_payload_key)
                graph_meta = st.session_state.get(graph_meta_key)

                if load_graphs_on_demand and st.session_state.get(graph_model_loaded_key) != selected_model:
                    st.info("Select a model and click `Load Selected Graph` to fetch and render that graph.")
                elif graph_json:
                    with st.expander(f"📊 {selected_model} - Architecture Graph", expanded=False):
                        # Display metadata
                        meta_col1, meta_col2, meta_col3 = st.columns(3)
                        if graph_meta:
                            meta_col1.metric("Original Size", f"{graph_meta.get('graph_size_original', 0)/1024:.1f} KB")
                            meta_col2.metric("Compressed Size", f"{graph_meta.get('graph_size_compressed', 0)/1024:.1f} KB")
                            compression = 100 * (1 - graph_meta.get('graph_size_compressed', 1) / max(graph_meta.get('graph_size_original', 1), 1))
                            meta_col3.metric("Compression", f"{compression:.0f}%")
                        
                        st.markdown("**Green** = Quantized Layers | **Gold** = Supported (Unquantized) | **Gray** = Structural/Other")
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
                            key=f"dl_json_{i}"
                        )
                else:
                    st.info(f"ℹ️ No architecture graph available for {selected_model}. "
                            f"Run `python runspace/src/database/generate_model_graphs.py` to generate graphs.")
        else:
            st.info("No data to display. Apply filters above to see models.")
        
        st.markdown("---")

    # Add Table Button
    st.button("➕ Add Table", on_click=add_table)

st.sidebar.markdown("---")
st.sidebar.info("Managed via `src/database/handler.py`")
