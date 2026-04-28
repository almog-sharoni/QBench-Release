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
from datetime import datetime

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from runspace.src.adapters.generic_adapter import GenericAdapter
    from runspace.src.utils.architecture_viz import generate_hierarchical_json
    GRAPH_VIZ_AVAILABLE = True
except Exception:
    GRAPH_VIZ_AVAILABLE = False

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

PRESETS_FILE = os.path.join(os.path.dirname(__file__), "presets.json")
DB_FOLDER = os.path.join(PROJECT_ROOT, "runspace/database")
DEFAULT_DB_PATH = os.path.join(DB_FOLDER, "runs.db")
DB_PATH = DEFAULT_DB_PATH
FM_DB_PATH = os.path.join(DB_FOLDER, "fm_runs.db")
RUN_WINDOW_TO_LIMIT = {
    "200 (fastest)": 200,
    "500": 500,
    "1000": 1000,
    "All": None,
}


def list_database_files():
    os.makedirs(DB_FOLDER, exist_ok=True)
    db_files = [
        name for name in os.listdir(DB_FOLDER)
        if name.endswith(".db") and os.path.isfile(os.path.join(DB_FOLDER, name))
    ]
    if not db_files:
        return [os.path.basename(DEFAULT_DB_PATH)]
    return sorted(db_files)


def make_safe_db_filename(name):
    raw_name = str(name or "").strip()
    safe_chars = []
    for char in raw_name:
        if char.isalnum() or char in ("-", "_", "."):
            safe_chars.append(char)
        elif char.isspace():
            safe_chars.append("_")
    safe_name = "".join(safe_chars).strip("._")
    if not safe_name:
        safe_name = f"selected_runs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not safe_name.endswith(".db"):
        safe_name += ".db"
    return safe_name


def resolve_db_path(db_name, sanitize=False):
    db_filename = make_safe_db_filename(db_name) if sanitize else os.path.basename(str(db_name or ""))
    if not db_filename.endswith(".db"):
        raise ValueError("Database filename must end with .db.")
    db_path = os.path.abspath(os.path.join(DB_FOLDER, db_filename))
    db_folder = os.path.abspath(DB_FOLDER)
    if os.path.commonpath([db_folder, db_path]) != db_folder:
        raise ValueError("Database path must stay inside runspace/database.")
    return db_path


def rename_database_file(current_db_name, new_db_name):
    current_path = resolve_db_path(current_db_name)
    new_safe_name = make_safe_db_filename(new_db_name)
    new_path = resolve_db_path(new_safe_name, sanitize=True)
    if not os.path.exists(current_path):
        raise FileNotFoundError(f"Database not found: {current_db_name}")
    if os.path.exists(new_path):
        raise FileExistsError(f"Database already exists: {new_safe_name}")
    os.rename(current_path, new_path)
    return new_safe_name


def delete_database_file(db_name):
    db_path = resolve_db_path(db_name)
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_name}")
    os.remove(db_path)


def choose_db_after_delete(deleted_db_name):
    remaining = [name for name in list_database_files() if name != deleted_db_name]
    default_db_name = os.path.basename(DEFAULT_DB_PATH)
    if default_db_name in remaining:
        return default_db_name
    return remaining[0] if remaining else default_db_name


def reset_filters_for_db_change():
    filter_key_prefixes = (
        "filter_",
        "table_",
        "name_input_",
    )
    for key in list(st.session_state.keys()):
        if key.startswith(filter_key_prefixes) or key == "dashboard_date_range":
            del st.session_state[key]
    st.session_state.num_tables = 1

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
    :root {
        --dashboard-app-bg:
            radial-gradient(circle at top left, rgba(20, 184, 166, 0.10), transparent 30%),
            radial-gradient(circle at top right, rgba(59, 130, 246, 0.12), transparent 28%),
            linear-gradient(180deg, #f8fbff 0%, #f4f7fb 100%);
        --dashboard-app-bg-color: #f4f7fb;
        --dashboard-sidebar-bg:
            linear-gradient(180deg, rgba(248, 250, 252, 0.97), rgba(241, 245, 249, 0.98));
        --dashboard-sidebar-bg-color: #f1f5f9;
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --dashboard-app-bg:
                radial-gradient(circle at top left, rgba(45, 212, 191, 0.14), transparent 34%),
                radial-gradient(circle at top right, rgba(59, 130, 246, 0.16), transparent 30%),
                linear-gradient(180deg, #020617 0%, #000000 100%);
            --dashboard-app-bg-color: #000000;
            --dashboard-sidebar-bg:
                linear-gradient(180deg, rgba(2, 6, 23, 0.98), rgba(0, 0, 0, 0.99));
            --dashboard-sidebar-bg-color: #000000;
        }
    }

    html[data-theme="dark"],
    body[data-theme="dark"],
    [data-theme="dark"] {
        --dashboard-app-bg:
            radial-gradient(circle at top left, rgba(45, 212, 191, 0.14), transparent 34%),
            radial-gradient(circle at top right, rgba(59, 130, 246, 0.16), transparent 30%),
            linear-gradient(180deg, #020617 0%, #000000 100%);
        --dashboard-app-bg-color: #000000;
        --dashboard-sidebar-bg:
            linear-gradient(180deg, rgba(2, 6, 23, 0.98), rgba(0, 0, 0, 0.99));
        --dashboard-sidebar-bg-color: #000000;
    }

    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background-color: var(--dashboard-app-bg-color);
    }

    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background: var(--dashboard-app-bg);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2.5rem;
    }

    .dashboard-hero {
        padding: 1.2rem 1.3rem;
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(15, 118, 110, 0.90));
        box-shadow: 0 18px 42px rgba(15, 23, 42, 0.14);
        color: #f8fafc;
        margin-bottom: 1rem;
    }

    .dashboard-hero__eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.76rem;
        font-weight: 700;
        color: rgba(244, 247, 251, 0.78);
        margin-bottom: 0.45rem;
    }

    .dashboard-hero h1 {
        margin: 0;
        font-size: 2rem;
        line-height: 1.05;
        letter-spacing: -0.03em;
    }

    .dashboard-hero p {
        margin: 0.7rem 0 0;
        max-width: 60rem;
        color: rgba(248, 250, 252, 0.86);
        font-size: 0.98rem;
    }

    .dashboard-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin: 0.45rem 0 0.2rem;
    }

    .dashboard-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.28);
        background: rgba(255, 255, 255, 0.74);
        color: #0f172a;
        padding: 0.35rem 0.7rem;
        font-size: 0.82rem;
        font-weight: 600;
    }

    .dashboard-chip--dark {
        background: rgba(255, 255, 255, 0.14);
        color: #f8fafc;
        border-color: rgba(255, 255, 255, 0.18);
    }

    .dashboard-section-title {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin: 0.3rem 0 0.65rem;
    }

    .dashboard-section-title h3 {
        margin: 0;
        font-size: 1.15rem;
        letter-spacing: -0.02em;
    }

    .dashboard-section-title p {
        margin: 0.2rem 0 0;
        color: #475569;
        font-size: 0.9rem;
    }

    .dashboard-selection-banner {
        border: 1px solid rgba(20, 184, 166, 0.24);
        background: linear-gradient(135deg, rgba(236, 253, 245, 0.98), rgba(239, 246, 255, 0.96));
        border-radius: 16px;
        padding: 0.8rem 0.95rem;
        margin: 0.7rem 0 1rem;
    }

    .dashboard-selection-banner strong {
        color: #0f172a;
    }

    .dashboard-selection-banner span {
        color: #475569;
        font-size: 0.9rem;
    }

    .dashboard-filter-note {
        color: #475569;
        font-size: 0.84rem;
        margin: 0.25rem 0 0.55rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(203, 213, 225, 0.9);
        background: rgba(255, 255, 255, 0.88);
        border-radius: 16px;
        padding: 0.25rem 0.2rem;
        box-shadow: 0 8px 24px rgba(148, 163, 184, 0.12);
    }

    div[data-testid="stMetricLabel"] {
        color: #475569;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: #0f172a;
    }

    .stButton > button {
        transition: background-color 0.1s ease, border 0.1s ease !important;
        border-radius: 10px !important;
    }

    div[data-testid="stHorizontalBlock"] .stButton > button {
        background-color: #ffffff !important;
        color: #334155 !important;
        border: 1px solid #cbd5e1 !important;
        padding: 0px 8px !important;
        min-height: 32px !important;
        height: 32px !important;
        font-size: 11px !important;
        line-height: normal !important;
        font-weight: 600;
        white-space: nowrap !important;
        width: 100% !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }

    div[data-testid="stHorizontalBlock"] .stButton > button[kind="primary"],
    div[data-testid="stHorizontalBlock"] .stButton > button[data-testid*="primary"] {
        background-color: #0f766e !important;
        color: white !important;
        border: 1px solid #0f766e !important;
    }

    div[data-testid="stHorizontalBlock"] {
        gap: 6px !important;
    }

    section[data-testid="stSidebar"] {
        background-color: var(--dashboard-sidebar-bg-color);
        background: var(--dashboard-sidebar-bg);
    }

    [data-testid="stCheckbox"] label[data-baseweb="checkbox"] > div:first-child {
        border-color: #0f766e !important;
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
    for p in ['uefp', 'ufp', 'efp', 'fp']:
        if parts[0].startswith(p):
            try: 
                bits = int(parts[0][len(p):])
                break
            except: pass
    else:
        if parts[0] == 'dyn':
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


def get_fm_runs(limit):
    if not os.path.exists(FM_DB_PATH):
        return pd.DataFrame()
    db = RunDatabase(db_path=FM_DB_PATH)
    return db.get_fm_runs(limit=limit)


def delete_runs_by_ids(run_ids):
    db = RunDatabase(db_path=DB_PATH)
    return db.delete_runs_by_ids(run_ids)


def update_experiment_type_by_ids(run_ids, experiment_type):
    db = RunDatabase(db_path=DB_PATH)
    return db.update_experiment_type_by_ids(run_ids, experiment_type)


def create_database_from_run_ids(run_ids, destination_db_path):
    db = RunDatabase(db_path=DB_PATH)
    return db.create_database_from_run_ids(run_ids, destination_db_path)


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
    for p in ["uefp", "ufp", "efp", "fp"]:
        if text.startswith(p):
            base = text.split("_", 1)[0]
            try:
                return int(base[len(p):])
            except Exception:
                continue
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


def render_dashboard_intro():
    st.markdown(
        """
        <div class="dashboard-hero">
            <h1>QBench Experiment Dashboard</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="QBench Experiment Dashboard", layout="wide")
inject_global_styles()
flash_message = st.session_state.pop("dashboard_flash_message", None)
if flash_message:
    st.success(flash_message)


@st.dialog("Delete Database?", width="small")
def show_delete_database_dialog(db_name):
    st.warning(f"You are about to permanently delete `{db_name}`.")
    st.caption("This removes the database file from `runspace/database` and cannot be undone.")
    col_cancel, col_delete = st.columns(2)
    if col_cancel.button("Cancel", key=f"cancel_delete_db_{db_name}", use_container_width=True):
        st.rerun()
    if col_delete.button("Yes, Delete DB", key=f"confirm_delete_db_modal_{db_name}", type="primary", use_container_width=True):
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
db_options = list_database_files()
default_db_name = os.path.basename(DEFAULT_DB_PATH)
pending_db_selection = st.session_state.pop("pending_selected_experiment_db", None)
if pending_db_selection in db_options:
    st.session_state["selected_experiment_db"] = pending_db_selection
default_db_index = db_options.index(default_db_name) if default_db_name in db_options else 0
selected_db_name = st.sidebar.selectbox(
    "Experiment DB",
    options=db_options,
    index=default_db_index,
    key="selected_experiment_db",
    on_change=reset_filters_for_db_change,
    help="Choose a SQLite database from runspace/database for the experiments table.",
)
DB_PATH = os.path.join(DB_FOLDER, selected_db_name)
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
    if st.button("Rename DB", key="rename_selected_db_btn", use_container_width=True):
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
        use_container_width=True,
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
st.sidebar.caption("Use `Refresh Data` after new experiments complete.")

st.markdown("---")

with st.expander(f"🔍 Feature Matching Runs (`fm_runs.db`)", expanded=False):
    fm_df = get_fm_runs(selected_run_limit)
    if fm_df.empty:
        st.info(f"No feature-matching runs found at `{FM_DB_PATH}`.")
    else:
        st.caption(f"Showing {len(fm_df)} FM runs from {FM_DB_PATH}.")

        fm_models = sorted(fm_df['model_name'].dropna().unique().tolist())
        fm_formats = _sort_quant_formats(fm_df['weight_dt'].dropna().unique().tolist())
        c1, c2 = st.columns(2)
        sel_models = c1.multiselect("Model", fm_models, default=fm_models, key="fm_model_filter")
        sel_formats = c2.multiselect("Weight format", fm_formats, default=fm_formats, key="fm_fmt_filter")
        fm_view = fm_df[fm_df['model_name'].isin(sel_models) & fm_df['weight_dt'].isin(sel_formats)].copy()

        # Δ vs reference for the matching metrics that have a ref_* counterpart.
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
        st.dataframe(fm_view[display_cols], use_container_width=True, hide_index=True)

        with st.expander("Show config_json for selected rows", expanded=False):
            for _, row in fm_view.iterrows():
                st.markdown(f"**[{row['id']}] {row['model_name']} — {row['weight_dt']}/{row['activation_dt']}**")
                st.json(_safe_json_load(row.get('config_json')))

st.markdown("---")

tab_exp, tab_cache, tab_graph = st.tabs(["📊 Experiments", "🗄️ Cache Simulation", "🏗️ Architecture Graph"])

# Renderer exposed by the else: block below; None if db is empty.
_graph_renderer_fn = None

# Enter the Experiments tab context without re-indenting the existing block.
tab_exp.__enter__()

with st.spinner("Loading experiment runs..."):
    df = get_runs(selected_run_limit)

if df.empty:
    st.warning("No runs found in the database yet. Run an experiment first!")
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
                use_container_width=True,
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
                    st.dataframe(chunk_df, use_container_width=True)

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
        st.dataframe(selected_df[existing_preview_cols], use_container_width=True, hide_index=True)

        if st.button(
            f"Save Experiment Name For {len(run_ids)} Row{'s' if len(run_ids) != 1 else ''}",
            key=f"confirm_edit_experiment_{table_index}",
            type="primary",
            use_container_width=True,
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
        st.dataframe(selected_df[existing_preview_cols], use_container_width=True, hide_index=True)

        if os.path.exists(destination_db_path):
            st.error("A database with this name already exists. Choose a new filename.")
            return

        if st.button(
            f"Create DB With {len(run_ids)} Row{'s' if len(run_ids) != 1 else ''}",
            key=f"confirm_create_db_{table_index}",
            type="primary",
            use_container_width=True,
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
            if btn_col1.button("📊 Load Selected Graph", key=f"load_graph_dialog_{table_index}", type="primary", use_container_width=True):
                should_load_now = True
            elif btn_col2.button("🔁 Regenerate", key=f"regen_graph_dialog_{table_index}", use_container_width=True):
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
        if action_col1.button("Select all", key=f"{session_key}_select_all", use_container_width=True):
            st.session_state[session_key] = options.copy()
            _sync_filter_checkboxes(session_key, options, st.session_state[session_key])
            st.rerun()
        if action_col2.button("Inverse", key=f"{session_key}_inverse", use_container_width=True):
            selected_set = set(st.session_state[session_key])
            st.session_state[session_key] = [opt for opt in options if opt not in selected_set]
            _sync_filter_checkboxes(session_key, options, st.session_state[session_key])
            st.rerun()
        if action_col3.button("Clear", key=f"{session_key}_clear", use_container_width=True):
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
    if st.sidebar.button("🔄 Refresh Data", type="primary", use_container_width=True):
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
                use_container_width=True,
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
                        use_container_width=True,
                    ):
                        if spec["key"] == f"btn_plot_{i}":
                            run_comparison = True
                        elif spec["handler"] is not None:
                            spec["handler"]()

                if run_comparison:
                    with st.spinner(f"Building comparison chart for {num_runs} runs..."):
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
        #     if graph_action_col.button("🏗️ Open Graph Viewer", key=f"open_graph_viewer_{i}", type="primary", use_container_width=True):
        #         show_graph_viewer(i, available_models)
        # else:
        #     st.info("No data to display. Apply filters above to see models.")
        
        # st.markdown("---")

    # Add Table Button
    st.button("➕ Add Table", on_click=add_table)

# Close the Experiments tab context.
tab_exp.__exit__(None, None, None)

# ── Cache Simulation Tab ──────────────────────────────────────────────────────
def _fmt_e(n):
    """Format element count for display."""
    try:
        n = int(n)
    except Exception:
        return str(n)
    if n >= 1_000_000:
        return f"{n/1_000_000:.3f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def _build_bank_states(layers, num_banks, bank_size, rule_meta=None):
    """
    Pre-compute the cache bank state *during* each layer's execution.

    rule_meta: optional dict of {rule_name: {'xin_from_cache': bool, 'permanents': str}}
               built from the simulation's rules_json.  When provided, xin visibility
               and weight-bank display are derived from rule properties instead of
               hardcoded rule names, so the viz stays correct as rules change.
    """
    import math

    # Build fast per-rule lookups.
    # xin_full:        True  → xin fully resident in cache during execution
    # wt_shows:        True  → weight banks are explicitly held during execution
    # pipeline_banks:  int   → extra boundary banks for xin-on-xout overlap rules (xin not shown separately)
    _xin_full       = {}
    _wt_shows       = {}
    _pipeline_banks = {}
    if rule_meta:
        for name, meta in rule_meta.items():
            _xin_full[name]       = bool(meta.get('xin_from_cache', False))
            _wt_shows[name]       = 'weight' in meta.get('permanents', '').lower()
            _pipeline_banks[name] = int(meta.get('pipeline_banks', 0))
    else:
        # Legacy fallback: covers both old (r1_/r2_/r3_) and new short names.
        for name in ('r1_global_fit', 'r2_residual', 'r2_pool',
                     'global_fit', 'residual', 'pool'):
            _xin_full[name] = True
            _wt_shows[name] = False
        for name in ('r2_conv_output_dominated', 'r2_conv_input_dominated',
                     'conv_output_dominated', 'conv_input_dominated',
                     'linear_stream_xout'):
            _xin_full[name]       = True
            _wt_shows[name]       = True
            _pipeline_banks[name] = 1
        for name in ('r2_stream_xin_keep_xout', 'stream_xin_keep_xout',
                     'r3_weights_plus_4banks', 'fallback'):
            _xin_full[name] = False
            _wt_shows[name] = True

    states = []
    for layer in layers:
        rule = layer.get('rule', '')
        stay = bool(layer.get('stay_on_chip', False))
        oe   = int(layer.get('output_elems', 0) or 0)
        we   = int(layer.get('weight_elems', 0) or 0)
        ie   = int(layer.get('input_elems',  0) or 0)

        ob = math.ceil(oe / bank_size) if oe > 0 else 0
        wb = math.ceil(we / bank_size) if we > 0 else 0
        ib = math.ceil(ie / bank_size) if ie > 0 else 0

        ob = min(ob, num_banks)
        wb = min(wb, num_banks)
        ib = min(ib, num_banks)

        xin_full      = _xin_full.get(rule, False)
        wt_shows      = _wt_shows.get(rule, False)
        pipeline_b    = _pipeline_banks.get(rule, 0)

        if pipeline_b > 0:
            # xin is written onto the dominant tensor's space; only pipeline boundary banks shown
            xin_b    = 0
            stream_b = pipeline_b
        elif xin_full:
            xin_b    = ib
            stream_b = 0
        else:
            xin_b    = 0
            stream_b = 2       # 2-bank streaming buffer for xin from external

        xr_b = 0
        layer_type = layer.get('type', '')
        if stay:
            if layer_type == 'Residual':
                # xin (skip) fully resident; xout + x_r each use 2 streaming banks
                xout_b, xr_b, wt_b = 2, 2, 0
            else:
                xout_b = ob
                if wt_shows:
                    wt_b = wb          # weights permanently resident (e.g. conv_output_dominated)
                elif we > 0:
                    wt_b = min(2, wb)  # 2 streaming banks for weight tiles (e.g. global_fit on Conv2d)
                else:
                    wt_b = 0           # no weights (pool)
        else:
            # off-chip: xout streamed out; weights may stay resident for streaming
            xout_b   = 0
            wt_b     = min(wb, num_banks - stream_b - 2) if wt_shows else 0
            stream_b = min(stream_b + 2, num_banks - wt_b)  # xin-stream + xout-stream

        used   = xin_b + xout_b + xr_b + wt_b + stream_b
        free_b = max(0, num_banks - used)

        # Build per-bank list
        banks = []
        for _ in range(xin_b):
            banks.append({'type': 'xin', 'label': 'xin'})
        for _ in range(xout_b):
            banks.append({'type': 'xout', 'label': 'xout'})
        for _ in range(xr_b):
            banks.append({'type': 'xr', 'label': 'xr'})
        for _ in range(wt_b):
            banks.append({'type': 'weights', 'label': 'W'})
        for _ in range(stream_b):
            banks.append({'type': 'stream', 'label': '~'})
        for _ in range(free_b):
            banks.append({'type': 'free', 'label': ''})
        while len(banks) < num_banks:
            banks.append({'type': 'free', 'label': ''})
        banks = banks[:num_banks]

        states.append({
            'name':     layer.get('name', ''),
            'type':     layer.get('type', ''),
            'rule':     rule,
            'reason':   layer.get('reason', ''),
            'stay':     stay,
            'xin_b':    xin_b,
            'xout_b':   xout_b,
            'xr_b':     xr_b,
            'wt_b':     wt_b,
            'stream_b': stream_b,
            'free_b':   free_b,
            'output':   _fmt_e(oe),
            'weights':  _fmt_e(we),
            'input':    _fmt_e(ie),
            'banks':    banks,
        })
    return states


def _render_bank_viz_html(states, num_banks, bank_size, cache_elements):
    """Return a self-contained HTML string for the interactive bank viewer."""
    states_json = json.dumps(states)
    bank_size_fmt = _fmt_e(bank_size)
    cache_fmt = _fmt_e(cache_elements)
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
          background: #f8fafc; color: #0f172a; padding: 14px; }}
  h3 {{ font-size: 13px; font-weight: 700; color: #475569; letter-spacing:.06em;
        text-transform: uppercase; margin-bottom: 10px; }}
  #controls {{ display:flex; align-items:center; gap:12px; margin-bottom:14px; flex-wrap:wrap; }}
  #layer-slider {{ flex:1; min-width:200px; accent-color:#0f766e; }}
  #layer-counter {{ font-size:12px; color:#64748b; white-space:nowrap; }}
  #nav-btns {{ display:flex; gap:6px; }}
  .nav-btn {{ cursor:pointer; background:#fff; border:1px solid #cbd5e1; border-radius:7px;
              color:#334155; padding:4px 10px; font-size:12px; font-weight:600;
              transition:background .1s; }}
  .nav-btn:hover {{ background:#f1f5f9; }}

  #info-panel {{ background:#fff; border:1px solid #e2e8f0; border-radius:12px;
                 padding:12px 14px; margin-bottom:14px; }}
  .info-row {{ display:flex; gap:20px; flex-wrap:wrap; margin-bottom:6px; }}
  .info-label {{ font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase;
                 letter-spacing:.05em; }}
  .info-val {{ font-size:13px; font-weight:600; color:#0f172a; }}
  .rule-pill {{ display:inline-block; background:#eff6ff; border:1px solid #bfdbfe;
                border-radius:999px; padding:2px 10px; font-size:11px; font-weight:700;
                color:#1d4ed8; }}
  .oncehip-pill {{ background:#f0fdf4; border-color:#86efac; color:#15803d; }}
  .offchip-pill  {{ background:#fef2f2; border-color:#fca5a5; color:#dc2626; }}

  #bank-container {{ display:flex; gap:4px; flex-wrap:nowrap; margin-bottom:8px; }}
  .bank {{ flex:1; min-width:0; border-radius:6px; display:flex; flex-direction:column;
           align-items:center; justify-content:center; padding:8px 2px; position:relative;
           transition: background .25s, border-color .25s; border:1.5px solid transparent;
           cursor:default; }}
  .bank:hover {{ border-color: #94a3b8 !important; }}
  .bank-num {{ font-size:9px; color:rgba(0,0,0,.35); font-weight:600; position:absolute;
               bottom:3px; }}
  .bank-lbl {{ font-size:10px; font-weight:700; color:rgba(0,0,0,.55); }}
  .bank.xin     {{ background:#fed7aa; }}
  .bank.xout    {{ background:#dcfce7; }}
  .bank.xr      {{ background:#f3e8ff; }}
  .bank.weights {{ background:#dbeafe; }}
  .bank.stream  {{ background:#fef9c3; }}
  .bank.free    {{ background:#f1f5f9; }}

  #legend {{ display:flex; gap:14px; flex-wrap:wrap; margin-bottom:12px; }}
  .leg-item {{ display:flex; align-items:center; gap:5px; font-size:11px; font-weight:600; color:#475569; }}
  .leg-dot {{ width:12px; height:12px; border-radius:3px; }}
  .leg-xin     {{ background:#fed7aa; border:1px solid #fb923c; }}
  .leg-xout    {{ background:#dcfce7; border:1px solid #86efac; }}
  .leg-xr      {{ background:#f3e8ff; border:1px solid #c084fc; }}
  .leg-weights {{ background:#dbeafe; border:1px solid #93c5fd; }}
  .leg-stream  {{ background:#fef9c3; border:1px solid #fde047; }}
  .leg-free    {{ background:#f1f5f9; border:1px solid #cbd5e1; }}

  #size-bar {{ display:flex; height:20px; border-radius:6px; overflow:hidden; margin-bottom:12px;
               border:1px solid #e2e8f0; }}
  .sz-seg {{ height:100%; display:flex; align-items:center; justify-content:center;
             font-size:10px; font-weight:700; color:rgba(0,0,0,.5); transition:width .3s; }}
  .sz-xin     {{ background:#fdba74; }}
  .sz-xout    {{ background:#bbf7d0; }}
  .sz-xr      {{ background:#e9d5ff; }}
  .sz-weights {{ background:#bfdbfe; }}
  .sz-stream  {{ background:#fef08a; }}
  .sz-free    {{ background:#f1f5f9; }}

  #stats {{ display:flex; gap:12px; flex-wrap:wrap; }}
  .stat-box {{ background:#fff; border:1px solid #e2e8f0; border-radius:10px;
               padding:8px 14px; min-width:80px; text-align:center; }}
  .stat-val {{ font-size:18px; font-weight:700; color:#0f172a; }}
  .stat-lbl {{ font-size:10px; color:#64748b; font-weight:600; text-transform:uppercase; }}
  .stat-box.orange {{ border-color:#fb923c; background:#fff7ed; }}
  .stat-box.green  {{ border-color:#86efac; background:#f0fdf4; }}
  .stat-box.blue   {{ border-color:#93c5fd; background:#eff6ff; }}
  .stat-box.yellow {{ border-color:#fde047; background:#fefce8; }}
  .stat-box.gray   {{ border-color:#cbd5e1; background:#f8fafc; }}
  .stat-box.red    {{ border-color:#fca5a5; background:#fef2f2; }}
</style>
</head>
<body>
<h3>Memory Banks — {num_banks} banks × {bank_size_fmt} elem = {cache_fmt} elem total</h3>

<div id="controls">
  <div id="nav-btns">
    <button class="nav-btn" onclick="step(-1)">◀ Prev</button>
    <button class="nav-btn" onclick="step(1)">Next ▶</button>
  </div>
  <input type="range" id="layer-slider" min="0" max="0" value="0" oninput="setLayer(+this.value)">
  <span id="layer-counter"></span>
</div>

<div id="info-panel">
  <div class="info-row">
    <div><div class="info-label">Layer</div><div class="info-val" id="i-name">—</div></div>
    <div><div class="info-label">Type</div><div class="info-val" id="i-type">—</div></div>
    <div><div class="info-label">xin</div><div class="info-val" id="i-input">—</div></div>
    <div><div class="info-label">xout</div><div class="info-val" id="i-output">—</div></div>
    <div><div class="info-label">Weights</div><div class="info-val" id="i-weights">—</div></div>
  </div>
  <div class="info-row">
    <div><div class="info-label">Rule</div><div class="info-val"><span class="rule-pill" id="i-rule">—</span></div></div>
    <div><div class="info-label">Decision</div><div class="info-val"><span class="rule-pill" id="i-stay">—</span></div></div>
    <div style="flex:1"><div class="info-label">Reason</div><div class="info-val" id="i-reason" style="font-size:12px;color:#475569">—</div></div>
  </div>
</div>

<div id="legend">
  <div class="leg-item"><div class="leg-dot leg-xin"></div> xin (in cache)</div>
  <div class="leg-item"><div class="leg-dot leg-xout"></div> xout (on-chip output)</div>
  <div class="leg-item"><div class="leg-dot leg-xr"></div> x_r (residual stream)</div>
  <div class="leg-item"><div class="leg-dot leg-weights"></div> Weights</div>
  <div class="leg-item"><div class="leg-dot leg-stream"></div> Streaming buffer (~)</div>
  <div class="leg-item"><div class="leg-dot leg-free"></div> Free</div>
</div>

<div id="size-bar">
  <div class="sz-seg sz-xin"     id="sz-xin"     style="width:0%"></div>
  <div class="sz-seg sz-xout"    id="sz-xout"    style="width:0%"></div>
  <div class="sz-seg sz-xr"      id="sz-xr"      style="width:0%"></div>
  <div class="sz-seg sz-weights" id="sz-weights" style="width:0%"></div>
  <div class="sz-seg sz-stream"  id="sz-stream"  style="width:0%"></div>
  <div class="sz-seg sz-free"    id="sz-free"    style="width:100%">free</div>
</div>

<div id="bank-container"></div>

<div id="stats">
  <div class="stat-box orange"><div class="stat-val" id="st-xin">0</div><div class="stat-lbl">xin banks</div></div>
  <div class="stat-box green" ><div class="stat-val" id="st-xout">0</div><div class="stat-lbl">xout banks</div></div>
  <div class="stat-box" id="xr-box" style="border-color:#c084fc;background:#faf5ff;display:none">
    <div class="stat-val" id="st-xr">0</div><div class="stat-lbl">x_r banks</div>
  </div>
  <div class="stat-box blue"  ><div class="stat-val" id="st-wt">0</div><div class="stat-lbl">weight banks</div></div>
  <div class="stat-box yellow" id="stream-box" style="display:none">
    <div class="stat-val" id="st-stream">0</div><div class="stat-lbl">stream buf</div>
  </div>
  <div class="stat-box gray"  ><div class="stat-val" id="st-free">0</div><div class="stat-lbl">free banks</div></div>
  <div class="stat-box red"   id="offchip-box" style="display:none">
    <div class="stat-val">OFF</div><div class="stat-lbl">xout → external</div>
  </div>
</div>

<script>
const STATES = {states_json};
const NUM_BANKS = {num_banks};

let current = 0;
const slider   = document.getElementById('layer-slider');
const counter  = document.getElementById('layer-counter');
const bankCont = document.getElementById('bank-container');

// Build bank DOM once
for (let i = 0; i < NUM_BANKS; i++) {{
  const b = document.createElement('div');
  b.className = 'bank free';
  b.id = 'bank-' + i;
  b.innerHTML = '<span class="bank-lbl" id="bl-' + i + '"></span><span class="bank-num">' + i + '</span>';
  bankCont.appendChild(b);
}}

slider.max = STATES.length - 1;

function step(d) {{ setLayer(Math.max(0, Math.min(STATES.length - 1, current + d))); }}

function setLayer(idx) {{
  current = idx;
  slider.value = idx;
  const s = STATES[idx];
  counter.textContent = (idx + 1) + ' / ' + STATES.length + ' — ' + s.name;

  document.getElementById('i-name').textContent    = s.name;
  document.getElementById('i-type').textContent    = s.type;
  document.getElementById('i-input').textContent   = s.input;
  document.getElementById('i-output').textContent  = s.output;
  document.getElementById('i-weights').textContent = s.weights;
  document.getElementById('i-rule').textContent    = s.rule || '—';
  document.getElementById('i-reason').textContent  = s.reason || '—';

  const stayEl = document.getElementById('i-stay');
  if (s.stay) {{
    stayEl.textContent = '✓ On-Chip';
    stayEl.className   = 'rule-pill oncehip-pill';
  }} else {{
    stayEl.textContent = '✗ Off-Chip';
    stayEl.className   = 'rule-pill offchip-pill';
  }}

  // Update banks
  for (let i = 0; i < NUM_BANKS; i++) {{
    const bank = s.banks[i] || {{type:'free', label:''}};
    const el   = document.getElementById('bank-' + i);
    const lbl  = document.getElementById('bl-' + i);
    el.className = 'bank ' + bank.type;
    lbl.textContent = bank.label;
  }}

  // Size bar
  function szSeg(id, banks, label) {{
    const pct = (banks / NUM_BANKS * 100).toFixed(1);
    const el = document.getElementById(id);
    el.style.width   = pct + '%';
    el.textContent   = pct > 5 ? (banks + 'B' + (label ? ' ' + label : '')) : '';
  }}
  szSeg('sz-xin',     s.xin_b,    'xin');
  szSeg('sz-xout',    s.xout_b,   'xout');
  szSeg('sz-xr',      s.xr_b,     'xr');
  szSeg('sz-weights', s.wt_b,     'W');
  szSeg('sz-stream',  s.stream_b, '~');
  szSeg('sz-free',    s.free_b,   'free');

  // Stats
  document.getElementById('st-xin').textContent  = s.xin_b;
  document.getElementById('st-xout').textContent = s.xout_b;
  document.getElementById('st-wt').textContent   = s.wt_b;
  document.getElementById('st-free').textContent = s.free_b;
  const stXr = document.getElementById('st-xr');
  if (stXr) stXr.textContent = s.xr_b;
  document.getElementById('xr-box').style.display     = (s.xr_b > 0) ? '' : 'none';
  const stEl = document.getElementById('st-stream');
  if (stEl) stEl.textContent = s.stream_b;
  document.getElementById('stream-box').style.display  = s.stream_b > 0 ? '' : 'none';
  document.getElementById('offchip-box').style.display = s.stay ? 'none' : '';
}}

setLayer(0);
</script>
</body>
</html>"""


with tab_cache:
    st.markdown("""
    <div class="dashboard-hero">
        <div class="dashboard-hero__eyebrow">ASIC · On-Chip Memory</div>
        <h1>Cache Simulation</h1>
        <p>Layer-by-layer cache placement decisions — which outputs stay on chip and which must be quantized for external memory transfer.</p>
    </div>
    """, unsafe_allow_html=True)

    @st.cache_data(ttl=30)
    def _load_cache_sims():
        from runspace.src.database.handler import RunDatabase
        db = RunDatabase(db_path=DB_PATH)
        return db.get_cache_simulations()

    cache_sim_df = _load_cache_sims()

    if cache_sim_df.empty:
        st.info(
            "No cache simulations in the database yet. "
            "Run `simulate_cache.py` to populate this tab."
        )
    else:
        # ── Model selector ────────────────────────────────────────────────────
        available_models = sorted(cache_sim_df['model_name'].dropna().unique())
        col_sel, col_refresh = st.columns([4, 1])
        selected_model = col_sel.selectbox(
            "Model", available_models, key="cache_sim_model_select"
        )
        if col_refresh.button("🔄 Refresh", key="cache_sim_refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

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
                use_container_width=True,
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

    rules_raw  = latest.get('rules_json') if not cache_sim_df.empty else None
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
            use_container_width=True,
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

# ── Architecture Graph Tab ───────────────────────────────────────────────────
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
            "Load Graph", key="graph_tab_generate", type="primary", use_container_width=True
        )
        regenerate_clicked = col_gr.button(
            "Regenerate", key="graph_tab_regenerate", use_container_width=True
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
