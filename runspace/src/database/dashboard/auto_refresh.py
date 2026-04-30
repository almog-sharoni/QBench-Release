"""Scoped dashboard refresh and right-side active-run monitor."""

import html as _dashboard_html
import json as _dashboard_json
import os as _dashboard_os
import streamlit.components.v1 as _dashboard_components


def _dashboard_right_panel_tail(path, max_lines=3):
    if not path or not _dashboard_os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-max_lines:]).strip()
    except Exception:
        return ""


def _dashboard_right_panel_run_title(run):
    command = str(run.get("command") or "")
    if "find_optimal_input_quant.py" in command:
        return "Input Quant"
    if "find_optimal_weight_quant.py" in command:
        return "Weight Quant"
    if "find_optimal_hybrid_quant.py" in command:
        return "Hybrid Quant"
    if "run_interactive.py" in command:
        return "Interactive Run"
    return "Dashboard Run"


def _dashboard_right_panel_progress(run):
    parser = globals().get("_dashboard_runner_parse_progress")
    log_path = run.get("log_path")
    if parser is None or not log_path or not _dashboard_os.path.exists(log_path):
        return 0, "Starting"
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            parsed = parser(f.read())
    except Exception:
        return 0, "Starting"

    if parsed.get("completed"):
        return 100, "Completed"

    total = parsed.get("total_configs") or 0
    current = parsed.get("current_config") or 0
    if total > 0:
        step_pct = (parsed.get("tqdm_percent") or 0) / 100.0
        pct = int(max(0, min(99, ((max(0, current - 1) + step_pct) / total) * 100)))
        label = f"Running config {current}/{total}"
        if parsed.get("current_name"):
            label += f" · {parsed['current_name']}"
        return pct, label

    tqdm_pct = parsed.get("tqdm_percent")
    if tqdm_pct is not None:
        return int(tqdm_pct), "Current step"
    return 8, "Starting"


def _dashboard_right_panel_css():
    return """
    :root {
        --qbench-right-sidebar-width: min(22rem, 25vw);
        --qbench-right-sidebar-rail: 3.45rem;
    }
    body.qbench-right-sidebar-open .block-container {
        padding-right: calc(var(--qbench-right-sidebar-width) + 3rem) !important;
        transition: padding-right 180ms ease;
    }
    body.qbench-right-sidebar-collapsed .block-container {
        padding-right: calc(var(--qbench-right-sidebar-rail) + 1.2rem) !important;
        transition: padding-right 180ms ease;
    }
    #qbench-right-runs-root { position: relative; z-index: 2147483000; }
    .qbench-right-runs {
        position: fixed;
        right: 0;
        top: 0;
        width: var(--qbench-right-sidebar-width);
        height: 100vh;
        max-height: 100vh;
        overflow-y: auto;
        z-index: 2147483000;
        border-left: 1px solid rgba(20, 184, 166, 0.28);
        border-radius: 0;
        background-color: #020617;
        background-image: radial-gradient(circle at 20% 0%, rgba(45, 212, 191, 0.16), transparent 30%), linear-gradient(180deg, #0f172a, #020617);
        color: #f8fafc;
        box-shadow: -18px 0 38px rgba(15, 23, 42, 0.22), -1px 0 0 rgba(255, 255, 255, 0.04);
        padding: 4.9rem 1.05rem 1.2rem;
        font-family: sans-serif;
        transition: width 180ms ease, padding 180ms ease;
        box-sizing: border-box;
        isolation: isolate;
    }
    .qbench-right-runs:not([open]) {
        width: var(--qbench-right-sidebar-rail);
        padding: 4.7rem 0.55rem 1rem;
        overflow: hidden;
    }
    .qbench-right-runs summary { cursor: pointer; list-style: none; outline: none; }
    .qbench-right-runs summary::-webkit-details-marker { display: none; }
    .qbench-right-runs[open] summary {
        position: sticky;
        top: 0;
        z-index: 1;
        margin: -4.9rem -1.05rem 1rem;
        padding: 1.15rem 1.05rem 1rem;
        background: #020617;
        border-bottom: 1px solid rgba(148, 163, 184, 0.18);
        color: #ccfbf1;
        font-weight: 850;
    }
    .qbench-right-runs:not([open]) summary {
        writing-mode: vertical-rl;
        text-orientation: mixed;
        white-space: nowrap;
        display: flex;
        align-items: center;
        gap: 0.45rem;
        color: #ccfbf1;
        font-weight: 800;
        min-height: calc(100vh - 5.7rem);
    }
    .qbench-right-runs:not([open]) .qbench-right-runs__body { display: none; }
    .qbench-right-runs h3 { margin: 0 0 0.25rem; font-size: 1rem; letter-spacing: -0.02em; }
    .qbench-right-runs__sub { color: rgba(226, 232, 240, 0.72); font-size: 0.78rem; margin-bottom: 0.95rem; }
    .qbench-run-card { border: 1px solid rgba(148, 163, 184, 0.22); border-radius: 16px; padding: 0.82rem; background: #0f172a; margin-bottom: 0.78rem; }
    .qbench-run-card--running { border-color: rgba(45, 212, 191, 0.42); box-shadow: inset 0 0 26px rgba(20, 184, 166, 0.08); }
    .qbench-run-card__top { display: flex; align-items: center; gap: 0.45rem; margin-bottom: 0.35rem; }
    .qbench-run-dot { width: 0.62rem; height: 0.62rem; border-radius: 999px; background: #5eead4; box-shadow: 0 0 0 5px rgba(94, 234, 212, 0.13), 0 0 20px rgba(94, 234, 212, 0.9); animation: qbenchPulse 1.35s ease-in-out infinite; }
    @keyframes qbenchPulse { 0%, 100% { opacity: 0.58; transform: scale(1); } 50% { opacity: 1; transform: scale(1.1); } }
    .qbench-run-progress-label { color: rgba(204, 251, 241, 0.95); font-size: 0.74rem; margin: 0.2rem 0 0.35rem; overflow-wrap: anywhere; }
    .qbench-run-progress { height: 0.55rem; border-radius: 999px; overflow: hidden; background: #020617; border: 1px solid rgba(255,255,255,0.12); margin-bottom: 0.55rem; }
    .qbench-run-progress span { display: block; height: 100%; min-width: 8%; border-radius: 999px; background: linear-gradient(90deg, #14b8a6, #67e8f9, #facc15); box-shadow: 0 0 20px rgba(103, 232, 249, 0.35); }
    .qbench-run-meta, .qbench-run-log { color: rgba(226, 232, 240, 0.72); font-size: 0.74rem; line-height: 1.25; overflow-wrap: anywhere; }
    .qbench-run-card pre { max-height: 5.6rem; overflow: hidden; margin: 0.55rem 0 0; white-space: pre-wrap; color: #ccfbf1; font-size: 0.68rem; line-height: 1.2; }
    .qbench-run-empty { border: 1px dashed rgba(148, 163, 184, 0.32); border-radius: 16px; padding: 0.85rem; color: rgba(226, 232, 240, 0.78); display: grid; gap: 0.24rem; margin-bottom: 0.95rem; background: #0f172a; }
    .qbench-run-empty span { font-size: 0.76rem; }
    .qbench-run-recent { border-top: 1px solid rgba(148, 163, 184, 0.18); padding-top: 0.7rem; margin-top: 0.3rem; }
    .qbench-run-recent h4 { margin: 0 0 0.45rem; font-size: 0.82rem; color: rgba(226, 232, 240, 0.82); }
    .qbench-run-recent ul { list-style: none; padding: 0; margin: 0; display: grid; gap: 0.38rem; }
    .qbench-run-recent li { display: flex; justify-content: space-between; gap: 0.5rem; color: rgba(248, 250, 252, 0.8); font-size: 0.72rem; }
    .qbench-run-recent em { color: rgba(148, 163, 184, 0.88); font-style: normal; white-space: nowrap; }
    @media (max-width: 1200px) {
        :root { --qbench-right-sidebar-width: min(19rem, 42vw); }
        body.qbench-right-sidebar-open .block-container {
            padding-right: calc(var(--qbench-right-sidebar-width) + 1.5rem) !important;
        }
    }
    @media (max-width: 760px) {
        :root { --qbench-right-sidebar-width: min(18rem, 78vw); }
        body.qbench-right-sidebar-open .block-container,
        body.qbench-right-sidebar-collapsed .block-container {
            padding-right: 1rem !important;
        }
        .qbench-right-runs { box-shadow: -18px 0 38px rgba(15, 23, 42, 0.34); }
    }
    """


@st.fragment(run_every=2)
def _render_dashboard_right_active_runs():
    refresh_registry = globals().get("_dashboard_runner_refresh_registry")
    if refresh_registry is None:
        return

    registry = refresh_registry()
    running = [run for run in registry if run.get("status") == "running"]
    recent = list(reversed(registry[-8:]))
    cards = []

    for run in running:
        title = _dashboard_html.escape(_dashboard_right_panel_run_title(run))
        pid = _dashboard_html.escape(str(run.get("pid", "-")))
        started = _dashboard_html.escape(str(run.get("start_time", "-")))
        log_name = _dashboard_html.escape(_dashboard_os.path.basename(str(run.get("log_path", ""))) or "-")
        tail = _dashboard_html.escape(_dashboard_right_panel_tail(run.get("log_path")))
        pct, progress_label = _dashboard_right_panel_progress(run)
        cards.append(f"""
            <div class="qbench-run-card qbench-run-card--running">
                <div class="qbench-run-card__top"><span class="qbench-run-dot"></span><strong>{title}</strong></div>
                <div class="qbench-run-progress-label">{_dashboard_html.escape(progress_label)}</div>
                <div class="qbench-run-progress"><span style="width:{pct}%"></span></div>
                <div class="qbench-run-meta">PID {pid} · {started}</div>
                <div class="qbench-run-log">{log_name}</div>
                <pre>{tail}</pre>
            </div>
        """)

    if not cards:
        cards.append('<div class="qbench-run-empty"><strong>No active runs</strong><span>Launch runs from the Run Models tab.</span></div>')

    recent_items = []
    for run in recent:
        status = _dashboard_html.escape(str(run.get("status", "-")))
        pid = _dashboard_html.escape(str(run.get("pid", "-")))
        title = _dashboard_html.escape(_dashboard_right_panel_run_title(run))
        recent_items.append(f"<li><span>{title}</span><em>{status} · {pid}</em></li>")
    recent_html = "".join(recent_items) if recent_items else "<li><span>No history yet</span><em>-</em></li>"
    open_attr = "open" if running else ""
    panel_html = f"""
    <details class="qbench-right-runs" data-default-open="{1 if running else 0}" {open_attr}>
        <summary>Active Runs · {len(running)}</summary>
        <div class="qbench-right-runs__body">
            <h3>Active Runs</h3>
            <div class="qbench-right-runs__sub">{len(running)} running · auto-updates · click header to collapse</div>
            {''.join(cards)}
            <div class="qbench-run-recent"><h4>Recent</h4><ul>{recent_html}</ul></div>
        </div>
    </details>
    """

    _dashboard_components.html(
        f"""
        <script>
        (() => {{
          const doc = window.parent.document;
          const css = {_dashboard_json.dumps(_dashboard_right_panel_css())};
          const html = {_dashboard_json.dumps(panel_html)};
          let style = doc.getElementById('qbench-right-runs-style');
          if (!style) {{
            style = doc.createElement('style');
            style.id = 'qbench-right-runs-style';
            doc.head.appendChild(style);
          }}
          style.textContent = css;

          let root = doc.getElementById('qbench-right-runs-root');
          if (!root) {{
            root = doc.createElement('div');
            root.id = 'qbench-right-runs-root';
            doc.body.appendChild(root);
          }}
          root.innerHTML = html;

          const panel = root.querySelector('.qbench-right-runs');
          if (!panel) return;
          const key = 'qbench-right-runs-open';
          const saved = window.localStorage.getItem(key);
          if (saved === '1') panel.open = true;
          if (saved === '0') panel.open = false;
          if (!saved && panel.dataset.defaultOpen === '1') panel.open = true;
          const syncLayout = () => {{
            doc.body.classList.toggle('qbench-right-sidebar-open', panel.open);
            doc.body.classList.toggle('qbench-right-sidebar-collapsed', !panel.open);
          }};
          syncLayout();
          panel.addEventListener('toggle', () => {{
            window.localStorage.setItem(key, panel.open ? '1' : '0');
            syncLayout();
          }});
        }})();
        </script>
        """,
        height=0,
    )


_render_dashboard_right_active_runs()
