import streamlit as st
import pandas as pd
import os
import sys
import base64

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

st.set_page_config(page_title="QBench Experiment Dashboard", layout="wide")

st.title("🚀 QBench Experiment Tracker")

# Initial Session State for Presets
if 'presets' not in st.session_state:
    st.session_state.presets = load_presets()

st.markdown("---")

# Initialize Database
db = RunDatabase()
df = db.get_runs()

if df.empty:
    st.warning("No runs found in the database yet. Run an experiment first!")
else:
    # Pre-process DF with parsed DT components
    for col in ['weight_dt', 'activation_dt']:
        prefix = 'w' if col.startswith('weight') else 'a'
        # Apply parsing and expand to separate columns
        parsed = df[col].apply(parse_dt)
        df[f'{prefix}_bits'] = parsed.apply(lambda x: x[0])
        df[f'{prefix}_exp'] = parsed.apply(lambda x: x[1])
        df[f'{prefix}_mant'] = parsed.apply(lambda x: x[2])
    @st.dialog("📈 Accuracy Comparison", width="large")
    def show_large_chart(chart_df):
        # Inject CSS to enable horizontal scroll specifically in the dialog (Modal)
        st.markdown("""
            <style>
            [data-testid="stDialog"] [data-testid="stVerticalBlock"] {
                overflow-x: auto !important;
                padding-bottom: 20px;
            }
            /* Target the chart container to ensure it doesn't squash */
            .element-container:has(iframe) {
                min-width: fit-content;
            }
            </style>
        """, unsafe_allow_html=True)
        
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
        # Persistent CSS injection on every rerun
        st.markdown("""
        <style>
        /* Smooth transitions for all buttons */
        .stButton > button { 
            transition: background-color 0.1s ease, border 0.1s ease !important; 
            border-radius: 4px !important;
        }
        
        /* DEFAULT: SOFT ROSE (for OFF state) */
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
        
        /* PRIMARY: SOFT EMERALD (for ON state) */
        div[data-testid="stHorizontalBlock"] .stButton > button[kind="primary"],
        div[data-testid="stHorizontalBlock"] .stButton > button[data-testid*="primary"] {
            background-color: #10b981 !important;
            color: white !important;
            border: 1px solid #059669 !important;
        }
        
        /* Tighten gap between columns in the horizontal block */
        div[data-testid="stHorizontalBlock"] {
            gap: 6px !important;
        }
        </style>
        """, unsafe_allow_html=True)

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

        # Bits are mandatory filters (all selected by default)
        filtered_df = filtered_df[filtered_df['w_bits'].isin(selected_w_bits)]
        filtered_df = filtered_df[filtered_df['a_bits'].isin(selected_a_bits)]

        # Exp/Mant are optional filters
        filtered_df = apply_opt_filter(filtered_df, 'w_exp', selected_w_exp, w_exp_options)
        filtered_df = apply_opt_filter(filtered_df, 'w_mant', selected_w_mant, w_mant_options)
        filtered_df = apply_opt_filter(filtered_df, 'a_exp', selected_a_exp, a_exp_options)
        filtered_df = apply_opt_filter(filtered_df, 'a_mant', selected_a_mant, a_mant_options)
        
        if not filtered_df.empty:
            # Calculate Accuracy Drop relative to reference
            if 'ref_acc1' in filtered_df.columns:
                filtered_df['acc1_drop'] = filtered_df['ref_acc1'] - filtered_df['acc1']
                filtered_df['acc5_drop'] = filtered_df['ref_acc5'] - filtered_df['acc5']
            if 'ref_certainty' in filtered_df.columns and 'certainty' in filtered_df.columns:
                filtered_df['cert_drop'] = filtered_df['ref_certainty'] - filtered_df['certainty']
            
            # Clean up and reorder columns for display
            cols_to_drop = ['run_date_dt', 'id']
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
            
            st.markdown(f"#### 📊 Visualization Options")
            # viz_all = st.checkbox("Visualize ALL filtered runs (ignores table selection)", key=f"viz_all_{i}")
            viz_all = False
            if viz_all or selected_indices:
                num_runs = len(display_df) if viz_all else len(selected_indices)
                st.success(f"Ready to visualize {num_runs} runs.")
                
                if st.button(f"📈 Generate Comparison ({num_runs})", key=f"btn_plot_{i}", type="primary", use_container_width=True):
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
                        chart_data.append({'Label': ref_label, 'MetricName': 'Acc1', 'MetricType': 'Reference (Acc1)', 'Accuracy (%)': first_row['ref_acc1'] if 'ref_acc1' in first_row else 0})
                        chart_data.append({'Label': ref_label, 'MetricName': 'Acc5', 'MetricType': 'Reference (Acc5)', 'Accuracy (%)': first_row['ref_acc5'] if 'ref_acc5' in first_row else 0})
                        
                        # 2. Add Quantized Entries
                        for _, row in model_rows.iterrows():
                            # Include date to make label unique if same format is run multiple times
                            # dt_str = pd.to_datetime(row['run_date']).strftime('%m/%d %H:%M')
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
        
        if not filtered_df.empty:
            # Get unique models in filtered data
            available_models = sorted(filtered_df['model_name'].unique())
            selected_model = st.selectbox(
                f"View quantization graph for (T{i+1})",
                options=available_models,
                key=f"graph_model_{i}"
            )
            
            if selected_model:
                # Try to get graph from database
                svg_content, metadata = db.get_model_graph_svg(selected_model)
                
                if svg_content:
                    with st.expander(f"📊 {selected_model} - Quantization Graph", expanded=False):
                        # Display metadata
                        meta_col1, meta_col2, meta_col3 = st.columns(3)
                        # graph_meta = db.get_model_graph_metadata(selected_model)
                        graph_meta = ''
                        if graph_meta:
                            meta_col1.metric("Original Size", f"{graph_meta.get('svg_size_original', 0)/1024:.1f} KB")
                            meta_col2.metric("Compressed Size", f"{graph_meta.get('svg_size_compressed', 0)/1024:.1f} KB")
                            compression = 100 * (1 - graph_meta.get('svg_size_compressed', 1) / max(graph_meta.get('svg_size_original', 1), 1))
                            meta_col3.metric("Compression", f"{compression:.0f}%")
                        
                        # Display interactive graph using streamlit-agraph
                        st.markdown("**Green** = Quantized Layers | **Gold** = Supported (Unquantized) | **Pink** = Unsupported | **Gray** = Structural/Other")
                        
                        st.info("💡 **Interactive**: Zoom & pan | Drag nodes | Scroll to zoom | Click to inspect")
                        
                        # Display interactive graph using pyvis
                        try:
                            import networkx as nx
                            from pyvis.network import Network
                            import xml.etree.ElementTree as ET
                            import tempfile
                            import os
                            import re
                            
                            # Parse SVG to extract exact nodes and edges
                            svg_no_ns = re.sub(r'\sxmlns=\"[^\"]+\"', '', svg_content, count=1)
                            root = ET.fromstring(svg_no_ns)
                            
                            G = nx.DiGraph()
                            
                            for g in root.iter('g'):
                                cls = g.get('class')
                                if cls == 'node':
                                    node_id = g.get('id', '')
                                    # Ignore legend objects to only show the model itself
                                    if node_id.startswith('legend_') or node_id.startswith('cluster_'):
                                        continue
                                        
                                    title_el = g.find('title')
                                    if title_el is None: continue
                                    title = title_el.text
                                    
                                    polygon = g.find('polygon')
                                    color = polygon.get('fill') if polygon is not None else '#D3D3D3'
                                    
                                    texts = [t.text for t in g.findall('text') if t.text]
                                    label = '\n'.join(texts) if texts else title
                                    
                                    G.add_node(title, label=label, color=color, shape='box')
                                    
                                elif cls == 'edge':
                                    title_el = g.find('title')
                                    if title_el is None: continue
                                    title = title_el.text
                                    
                                    if '->' in title:
                                        src, dst = title.split('->', 1)
                                        if src in G and dst in G:
                                            G.add_edge(src, dst)
                                    elif '&#45;&gt;' in title:
                                        src, dst = title.split('&#45;&gt;', 1)
                                        if src in G and dst in G:
                                            G.add_edge(src, dst)

                            if G.number_of_nodes() > 0:
                                net = Network(
                                    height='700px',
                                    width='100%',
                                    directed=True,
                                    notebook=True,
                                    cdn_resources='in_line'
                                )
                                net.from_nx(G)
                                
                                # Use hierarchical layout to look like SVG, but allow drag/zoom
                                net.set_options("""
                                var options = {
                                  "layout": {
                                    "hierarchical": {
                                      "enabled": true,
                                      "direction": "UD",
                                      "sortMethod": "directed",
                                      "levelSeparation": 250,
                                      "nodeSpacing": 250,
                                      "treeSpacing": 250
                                    }
                                  },
                                  "physics": {
                                    "enabled": false
                                  },
                                  "interaction": {
                                    "dragNodes": true,
                                    "dragView": true,
                                    "zoomView": true
                                  }
                                }
                                """)
                                
                                temp_file = os.path.join(tempfile.gettempdir(), f'graph_{selected_model}.html')
                                net.write_html(temp_file)
                                
                                with open(temp_file, 'r', encoding='utf-8') as f:
                                    html_content = f.read()
                                
                                # Inject Reset View button overlay into the generated pyvis HTML
                                reset_btn = '''<button onclick="network.fit({animation:{duration:500}});" style="position: absolute; bottom: 20px; right: 20px; z-index: 1000; padding: 10px 16px; background-color: #ffffff; border: 1px solid #d1d5db; border-radius: 6px; cursor: pointer; font-family: sans-serif; font-size: 14px; font-weight: 600; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); color: #374151; transition: all 0.2s;">🎯 Reset View</button>'''
                                html_content = html_content.replace('<body>', f'<body>\n{reset_btn}')
                                
                                import streamlit.components.v1 as components
                                components.html(html_content, height=750)
                            else:
                                st.warning("Could not extract nodes from graph SVG.")
                                

                        except Exception as e:
                            import traceback
                            traceback.print_exc()
                            st.error(f"Error: {str(e)}")
                            # Fallback: Display SVG as static image
                            svg_b64 = base64.b64encode(svg_content.encode('utf-8')).decode()
                            st.markdown(
                                f'<img src="data:image/svg+xml;base64,{svg_b64}" style="max-width:100%; height:auto; border: 1px solid #ddd; border-radius: 8px; padding: 10px; background: white;">',
                                unsafe_allow_html=True
                            )
                        
                        # Download button
                        st.download_button(
                            label="📥 Download SVG",
                            data=svg_content,
                            file_name=f"{selected_model}_graph.svg",
                            mime="image/svg+xml"
                        )
                        
                        # Show metadata details
                        if False:
                            with st.expander("Graph Metadata"):
                                st.json(metadata)
                else:
                    st.info(f"ℹ️ No quantization graph available for {selected_model}. "
                            f"Run `python runspace/src/database/generate_model_graphs.py` to generate graphs.")
        else:
            st.info("No data to display. Apply filters above to see models.")
        
        st.markdown("---")

    # Add Table Button
    st.button("➕ Add Table", on_click=add_table)

st.sidebar.markdown("---")
st.sidebar.info("Managed via `src/database/handler.py`")
