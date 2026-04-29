"""Streamlit dashboard entry point.

The dashboard implementation is split into ordered files under
`runspace/src/database/dashboard/`. They are executed in this module's global
namespace to preserve the original single-file behavior, especially Streamlit
callbacks and runtime globals such as `DB_PATH`.
"""

import os

_DASHBOARD_PARTS = [
    "setup.py",
    "styles.py",
    "data_helpers.py",
    "graph_helpers.py",
    "experiments_tab.py",
    "cache_helpers.py",
    "cache_tab.py",
    "run_models_tab.py",
    "graph_tab.py",
]


def _run_dashboard_parts():
    dashboard_dir = os.path.join(os.path.dirname(__file__), "dashboard")
    namespace = globals()
    for part in _DASHBOARD_PARTS:
        part_path = os.path.join(dashboard_dir, part)
        with open(part_path, "r", encoding="utf-8") as part_file:
            exec(compile(part_file.read(), part_path, "exec"), namespace)


_run_dashboard_parts()
