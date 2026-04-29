import streamlit.components.v1 as _dashboard_components

if st.session_state.get("dashboard_auto_refresh_enabled", True):
    interval = int(st.session_state.get("dashboard_auto_refresh_interval", 30) or 30)
    _dashboard_components.html(
        f"""
        <script>
        (() => {{
          const intervalMs = {max(10, interval) * 1000};
          const key = "qbench-dashboard-auto-refresh";
          if (window.parent[key]) {{
            window.parent.clearTimeout(window.parent[key]);
          }}
          window.parent[key] = window.parent.setTimeout(() => {{
            window.parent.location.reload();
          }}, intervalMs);
        }})();
        </script>
        """,
        height=0,
    )
