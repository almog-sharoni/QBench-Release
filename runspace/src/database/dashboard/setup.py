import streamlit as st
import pandas as pd
import os
import sys
import sqlite3

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


def _table_row_count(db_path, table_name):
    if not os.path.exists(db_path):
        return 0
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            )
            if cursor.fetchone() is None:
                return 0
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            return int(cursor.fetchone()[0] or 0)
    except sqlite3.Error:
        return 0


def list_database_files_for_run_kind(run_kind):
    db_files = list_database_files()
    if run_kind == "feature_matching":
        fm_files = [
            name for name in db_files
            if name.startswith("fm") or _table_row_count(os.path.join(DB_FOLDER, name), "fm_runs") > 0
        ]
        return fm_files or [os.path.basename(FM_DB_PATH)]

    cls_files = [
        name for name in db_files
        if not name.startswith("fm") or _table_row_count(os.path.join(DB_FOLDER, name), "runs") > 0
    ]
    return cls_files or [os.path.basename(DEFAULT_DB_PATH)]


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

