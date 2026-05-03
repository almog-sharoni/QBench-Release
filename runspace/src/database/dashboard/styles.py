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

    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 3rem !important;
        pointer-events: auto !important;
    }

    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    [data-testid="stDeployButton"],
    .stDeployButton,
    #MainMenu,
    footer {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }

    [data-testid="stToolbar"] {
        visibility: visible !important;
        pointer-events: auto !important;
    }

    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"],
    [data-testid="baseButton-headerNoPadding"],
    button[title*="sidebar"],
    button[aria-label*="sidebar"],
    button[title*="Sidebar"],
    button[aria-label*="Sidebar"],
    button[kind="header"] {
        display: flex !important;
        visibility: visible !important;
        pointer-events: auto !important;
        z-index: 999999 !important;
    }

    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background: var(--dashboard-app-bg);
    }

    .block-container {
        padding-top: 1rem;
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

    /* ── Suppress fragment auto-refresh "breathing" animation ──────────────
       Streamlit temporarily sets opacity: 0.33 + a transition on fragment
       containers whenever run_every fires. This makes content pulse white.
       We lock opacity to 1 and kill the transition to eliminate the flash.
    */
    [data-testid="stVerticalBlock"],
    [data-testid="stVerticalBlockBorderWrapper"],
    [data-testid="stHorizontalBlock"] {
        opacity: 1 !important;
        transition: none !important;
    }

    /* Also override the CSS custom property Streamlit uses in some versions */
    * {
        --stale-opacity: 1 !important;
        --fragment-transition-duration: 0s !important;
    }

    /* Kill any keyframe pulse/shimmer that Streamlit injects */
    @keyframes pulse {
        0%, 50%, 100% { opacity: 1; }
    }
    @keyframes streamlitPulse {
        0%, 50%, 100% { opacity: 1; }
    }
    @keyframes placeholderShimmer {
        0%, 100% { background-position: 0 0; }
    }

    /* Re-enable the intentional button hover transition (specificity wins) */
    .stButton > button {
        transition: background-color 0.1s ease, border-color 0.1s ease !important;
    }
    </style>
    """, unsafe_allow_html=True)
