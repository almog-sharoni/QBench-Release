"""Dashboard data refresh hook.

The dashboard intentionally does not force browser-level reloads here. Full
page reloads steal focus from tables, filters, editors, and log viewers. Data
is fetched from SQLite on Streamlit reruns, and active run logs have their own
scoped refresh loop in the Run Models tab.
"""
