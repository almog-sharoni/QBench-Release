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
                        # This prevents Streamlit from crashing if a model/format was removed from DB
                        valid_val = [v for v in val if v in options]
                        st.session_state[session_key] = valid_val
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
            selected_models = st.multiselect(f"Models (T{i+1})", options=models, default=models, key=f"filter_m_{i}")
        with col_f2:
            selected_exprs = st.multiselect(f"Experiment Types (T{i+1})", options=expr_types, default=expr_types, key=f"filter_e_{i}")
        
        # Advanced DT Filtering
        with st.expander(f"Advanced Datatype Filters (T{i+1})"):
            col_w_bits, col_w_exp, col_w_mant = st.columns(3)
            with col_w_bits:
                w_bits_options = sorted(df['w_bits'].dropna().unique())
                selected_w_bits = st.multiselect(
                    "Weight Bits", 
                    options=w_bits_options, 
                    default=w_bits_options, 
                    key=f"filter_wb_{i}",
                    format_func=lambda x: "Dynamic" if x == 0 else f"{int(x)} Bits"
                )
            with col_w_exp:
                w_exp_options = sorted(df['w_exp'].dropna().unique())
                selected_w_exp = st.multiselect("Weight Exponent", options=w_exp_options, default=w_exp_options, key=f"filter_we_{i}")
            with col_w_mant:
                w_mant_options = sorted(df['w_mant'].dropna().unique())
                selected_w_mant = st.multiselect("Weight Mantissa", options=w_mant_options, default=w_mant_options, key=f"filter_wm_{i}")

            col_a_bits, col_a_exp, col_a_mant = st.columns(3)
            with col_a_bits:
                a_bits_options = sorted(df['a_bits'].dropna().unique())
                selected_a_bits = st.multiselect(
                    "Activation Bits", 
                    options=a_bits_options, 
                    default=a_bits_options, 
                    key=f"filter_ab_{i}",
                    format_func=lambda x: "Dynamic" if x == 0 else f"{int(x)} Bits"
                )
            with col_a_exp:
                a_exp_options = sorted(df['a_exp'].dropna().unique())
                selected_a_exp = st.multiselect("Activation Exponent", options=a_exp_options, default=a_exp_options, key=f"filter_ae_{i}")
            with col_a_mant:
                a_mant_options = sorted(df['a_mant'].dropna().unique())
                selected_a_mant = st.multiselect("Activation Mantissa", options=a_mant_options, default=a_mant_options, key=f"filter_am_{i}")

            # Save Preset UI under each table's filters
            with st.expander(f"💾 Save current filters as Preset (T{i+1})"):
                p_col1, p_col2 = st.columns([3, 1])
                preset_name = p_col1.text_input("Preset Name", key=f"name_input_{i}", placeholder="e.g. My Optimal ResNet")
                if p_col2.button("Save", key=f"save_btn_{i}", use_container_width=True):
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
                
                if st.button(f"📈 Generate  Comparison ({num_runs})", key=f"btn_plot_{i}"):
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

    # Add Table Button
    st.button("➕ Add Table", on_click=add_table)

st.sidebar.markdown("---")
st.sidebar.info("Managed via `src/database/handler.py`")
