# Dashboard File Map

`runspace/src/database/dashboard.py` is now a small loader that executes these copied sections in order. The sections share one global namespace, matching the old single-file Streamlit behavior.

| File | Original lines | Purpose |
| --- | ---: | --- |
| `setup.py` | 1-139 | Imports, project path setup, constants, database file helpers, preset helpers. |
| `styles.py` | 140-363 | Global Streamlit/CSS styling. |
| `data_helpers.py` | 364-685 | Datatype parsing, run loading/mutation helpers, dataframe preprocessing, quantization win-rate calculations. |
| `graph_helpers.py` | 686-879 | Architecture graph generation/cache helpers, win-rate renderers, dashboard intro helper. |
| `experiments_tab.py` | 880-3002 | Main Streamlit setup, sidebar controls, feature-matching section, experiments tab and dialogs. |
| `cache_helpers.py` | 3003-3385 | Cache simulation formatting and bank visualization helper functions. |
| `cache_tab.py` | 3386-3583 | Cache Simulation tab UI. |
| `run_models_tab.py` | new | Dashboard launcher for non-interactive `run_interactive.py` model runs. |
| `graph_tab.py` | 3584-3650 | Architecture Graph tab UI and sidebar footer. |
