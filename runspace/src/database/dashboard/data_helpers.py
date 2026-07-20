def parse_dt(dt_str):
    """Extract (bits, exp, mant) from DT strings like 'fp4_e1m2'."""
    if not isinstance(dt_str, str) or not dt_str:
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


@st.cache_data(ttl=30, show_spinner=False)
def get_runs(db_path, limit):
    db = RunDatabase(db_path=db_path)
    return db.get_runs(limit=limit)


@st.cache_data(ttl=30, show_spinner=False)
def get_fm_runs(fm_db_path, limit):
    if not os.path.exists(fm_db_path):
        return pd.DataFrame()
    db = RunDatabase(db_path=fm_db_path)
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


BASELINE_ACCURACY_SERIES_ORDER = [
    "Weight opt",
    "Weight baseline",
    "Input dynamic MSE",
    "Input dynamic L1",
    "Input dynamic",
    "Input baseline",
]
BASELINE_ACCURACY_CHART_ORDER = ["Weights", "Inputs"]
HYBRID_ACCURACY_SERIES_ORDER = ["Hybrid fixed", "Hybrid dynamic"]


def _extract_trailing_bit_number(text):
    if text is None:
        return None

    normalized = str(text).strip().lower().replace("-", "_")
    for token in reversed(normalized.split("_")):
        token = token.strip()
        if token.endswith("bits"):
            token = token[:-4]
        elif token.endswith("bit"):
            token = token[:-3]
        if token.isdigit():
            return int(token)
    return None


def _coerce_positive_bit(value):
    try:
        if pd.isna(value):
            return None
        bit_value = int(float(value))
    except Exception:
        return None
    return bit_value if bit_value > 0 else None


def _dashboard_text(value, default=""):
    """Coerce nullable pandas scalar text without evaluating ``pd.NA``."""
    if value is None:
        return default
    try:
        if bool(pd.isna(value)):
            return default
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text else default


def _extract_embedded_format_bit_number(text):
    """Return a bit width embedded in a token such as ``fp4`` or ``ufp6``."""
    if not text:
        return None
    normalized = str(text).strip().lower().replace("-", "_")
    for token in normalized.split("_"):
        bit_number, _, _ = parse_dt(token)
        if bit_number is not None and bit_number > 0:
            return int(bit_number)
    return None


def _extract_max_format_bit_from_json(value):
    """Find the largest explicit fp/ufp format width in a serialized map."""
    if not value:
        return None
    try:
        payload = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError, json.JSONDecodeError):
        return None

    bit_numbers = []

    def _visit(item):
        if isinstance(item, str):
            bit_number, _, _ = parse_dt(item)
            if bit_number is not None and bit_number > 0:
                bit_numbers.append(int(bit_number))
        elif isinstance(item, dict):
            for nested_value in item.values():
                _visit(nested_value)
        elif isinstance(item, (list, tuple)):
            for nested_value in item:
                _visit(nested_value)

    _visit(payload)
    return max(bit_numbers) if bit_numbers else None


def _extract_dynamic_candidate_ceiling(config_value):
    """Read the configured maximum dynamic candidate width without treating fp32 runtime fields as candidates."""
    if not config_value:
        return None
    try:
        payload = json.loads(config_value) if isinstance(config_value, str) else config_value
    except (TypeError, ValueError, json.JSONDecodeError):
        return None

    candidate_bits = []

    def _visit(item):
        if isinstance(item, dict):
            for key, nested_value in item.items():
                if key == "candidate_formats" and isinstance(nested_value, (list, tuple)):
                    for format_name in nested_value:
                        bit_number, _, _ = parse_dt(format_name)
                        if bit_number is not None and bit_number > 0:
                            candidate_bits.append(int(bit_number))
                else:
                    _visit(nested_value)
        elif isinstance(item, (list, tuple)):
            for nested_value in item:
                _visit(nested_value)

    _visit(payload)
    return max(candidate_bits) if candidate_bits else None


def _baseline_accuracy_series_and_bits(row):
    exp_type = str(row.get("experiment_type", "") or "").strip().lower()

    if "_ablation_" in exp_type:
        return None, None, None

    if exp_type.startswith("weight_quant_optimized"):
        bit_number = _extract_trailing_bit_number(exp_type)
        if bit_number is None:
            bit_number = _coerce_positive_bit(row.get("w_bits"))
        if bit_number is None:
            bit_number = _extract_max_format_bit_from_json(row.get("quant_map_json"))
        return "Weights", "Weight opt", bit_number

    if exp_type.startswith("input_quant_dynamic"):
        activation_dt = _dashboard_text(row.get("activation_dt")).lower()
        if activation_dt.startswith("dyn_input_l1"):
            series_name = "Input dynamic L1"
        elif activation_dt.startswith("dyn_input_mse"):
            series_name = "Input dynamic MSE"
        else:
            config = _safe_json_load(row.get("config_json"))
            config = config if isinstance(config, dict) else {}
            evaluation = config.get("evaluation", {})
            evaluation = evaluation if isinstance(evaluation, dict) else {}
            input_quant = evaluation.get("dynamic_input_quant") or evaluation.get(
                "input_quant"
            )
            input_quant = input_quant if isinstance(input_quant, dict) else {}
            metric = str(input_quant.get("metric", "") or "").strip().lower()
            if metric == "l1":
                series_name = "Input dynamic L1"
            elif metric in {"mse", "l2"}:
                series_name = "Input dynamic MSE"
            else:
                series_name = "Input dynamic"
        bit_number = _extract_trailing_bit_number(exp_type)
        if bit_number is None:
            bit_number = _extract_embedded_format_bit_number(exp_type)
        if bit_number is None:
            bit_number = _coerce_positive_bit(row.get("a_bits"))
        if bit_number is None:
            bit_number = _extract_max_format_bit_from_json(row.get("input_map_json"))
        if bit_number is None:
            bit_number = _extract_dynamic_candidate_ceiling(row.get("config_json"))
        return "Inputs", series_name, bit_number

    if exp_type.startswith("input_quant_baseline"):
        return "Inputs", "Input baseline", _coerce_positive_bit(row.get("a_bits"))

    if exp_type.startswith("weight_quant_baseline"):
        return "Weights", "Weight baseline", _coerce_positive_bit(row.get("w_bits"))

    return None, None, None


def build_baseline_accuracy_plot_df(df):
    if df is None or df.empty:
        return pd.DataFrame()

    working_df = df.copy()
    if "w_bits" not in working_df.columns or "a_bits" not in working_df.columns:
        working_df = preprocess_runs_df(working_df)
    if working_df is None or working_df.empty or "acc1" not in working_df.columns:
        return pd.DataFrame()

    if "status" in working_df.columns:
        status_text = working_df["status"].fillna("").astype(str).str.upper()
        working_df = working_df[(status_text == "") | (status_text == "SUCCESS")]

    working_df["acc1_numeric"] = pd.to_numeric(working_df["acc1"], errors="coerce")
    working_df = working_df.dropna(subset=["acc1_numeric"])
    if working_df.empty:
        return pd.DataFrame()

    rows = []
    for _, run_row in working_df.iterrows():
        chart_kind, series_name, bit_number = _baseline_accuracy_series_and_bits(run_row)
        if series_name is None or bit_number is None:
            continue
        weight_dt = str(run_row.get("weight_dt", "") or "")
        activation_dt = str(run_row.get("activation_dt", "") or "")
        variant = weight_dt if chart_kind == "Weights" else activation_dt
        is_baseline = "baseline" in series_name.lower()
        _, exponent_bits, _ = parse_dt(variant)

        rows.append({
            "Model": str(run_row.get("model_name", "unknown") or "unknown"),
            "Chart": chart_kind,
            "Series": series_name,
            "Bits": int(bit_number),
            "Accuracy (%)": float(run_row["acc1_numeric"]),
            "Variant": variant,
            "Format Label": variant if is_baseline else series_name,
            "Exponent Family": f"e{exponent_bits}" if exponent_bits is not None else None,
            "Weight DT": weight_dt,
            "Activation DT": activation_dt,
            "Experiment Type": str(run_row.get("experiment_type", "") or ""),
            "Run Date": str(run_row.get("run_date", "") or ""),
            "Run Date Sort": pd.to_datetime(run_row.get("run_date"), errors="coerce"),
            "Run ID": run_row.get("id", None),
        })

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return plot_df

    plot_df["Series"] = pd.Categorical(
        plot_df["Series"],
        categories=BASELINE_ACCURACY_SERIES_ORDER,
        ordered=True,
    )
    plot_df["Chart"] = pd.Categorical(
        plot_df["Chart"],
        categories=BASELINE_ACCURACY_CHART_ORDER,
        ordered=True,
    )
    plot_df["Run ID Sort"] = pd.to_numeric(plot_df["Run ID"], errors="coerce").fillna(-1)
    plot_df = plot_df.sort_values(
        by=["Model", "Chart", "Bits", "Series", "Variant", "Run Date Sort", "Run ID Sort"],
        ascending=[True, True, True, True, True, False, False],
    )
    return plot_df.drop(columns=["Run Date Sort", "Run ID Sort"])


def _hybrid_accuracy_metadata(row):
    """Resolve hybrid run mode, activation width, and logged sweep metadata."""
    config = _safe_json_load(row.get("config_json"))
    config = config if isinstance(config, dict) else {}
    experiment = config.get("experiment", {})
    experiment = experiment if isinstance(experiment, dict) else {}

    experiment_type = _dashboard_text(row.get("experiment_type"))
    experiment_name = _dashboard_text(experiment.get("name"))
    if not (
        experiment_type.lower().startswith("hybrid")
        or experiment_name.lower().startswith("hybrid")
    ):
        return None

    evaluation = config.get("evaluation", {})
    evaluation = evaluation if isinstance(evaluation, dict) else {}
    input_quant = evaluation.get("input_quant") or evaluation.get(
        "dynamic_input_quant"
    )
    input_quant = input_quant if isinstance(input_quant, dict) else {}

    meta = config.get("meta", {})
    meta = meta if isinstance(meta, dict) else {}
    selection = meta.get("selection", {})
    selection = selection if isinstance(selection, dict) else {}
    weight_selection = selection.get("weight", {})
    weight_selection = (
        weight_selection if isinstance(weight_selection, dict) else {}
    )
    input_selection = selection.get("input", {})
    input_selection = input_selection if isinstance(input_selection, dict) else {}

    activation_dt = _dashboard_text(row.get("activation_dt"))
    logged_mode = _dashboard_text(
        input_selection.get("mode") or input_quant.get("mode")
    ).lower()
    is_dynamic = activation_dt.lower().startswith("dyn") or logged_mode == "dynamic"
    series_name = "Hybrid dynamic" if is_dynamic else "Hybrid fixed"

    if activation_dt.lower() == "fp32" and not is_dynamic:
        return None

    bit_number = _coerce_positive_bit(input_selection.get("bit_width"))
    if bit_number is None:
        bit_number = _extract_trailing_bit_number(activation_dt)
    if bit_number is None:
        bit_number = _extract_embedded_format_bit_number(activation_dt)
    if bit_number is None:
        bit_number = _coerce_positive_bit(row.get("a_bits"))
    if bit_number is None:
        bit_number = _extract_max_format_bit_from_json(row.get("input_map_json"))
    if bit_number is None:
        bit_number = _extract_dynamic_candidate_ceiling(config)
    if bit_number is None:
        return None

    candidates = input_quant.get("candidate_formats") or input_selection.get(
        "candidate_formats"
    )
    candidate_label = (
        ", ".join(str(fmt) for fmt in candidates)
        if isinstance(candidates, (list, tuple))
        else ""
    )

    max_batches = evaluation.get("max_batches")
    if max_batches is None:
        dataset = config.get("dataset", {})
        dataset = dataset if isinstance(dataset, dict) else {}
        max_batches = dataset.get("limit_batches")
    scope_known = max_batches is not None
    try:
        full_evaluation = scope_known and int(max_batches) <= 0
    except (TypeError, ValueError):
        full_evaluation = False
        scope_known = False
    if not scope_known:
        evaluation_scope = "Unknown"
    elif full_evaluation:
        evaluation_scope = "Full dataset"
    else:
        evaluation_scope = (
            f"{max_batches} batch{'es' if str(max_batches) != '1' else ''}"
        )

    legacy_sweep_entry = (
        str(weight_selection.get("mode", "")).lower() == "best_baseline"
        and str(input_selection.get("mode", "")).lower()
        in {"best_baseline", "dynamic"}
    )
    bidirectional_sweep_entry = str(selection.get("direction", "")).lower() in {
        "weight_fixed",
        "input_fixed",
    }
    sweep_entry = legacy_sweep_entry or bidirectional_sweep_entry
    sweep_direction = str(selection.get("direction", "")).strip().lower()

    weight_bit_number = _coerce_positive_bit(weight_selection.get("bit_width"))
    if weight_bit_number is None:
        weight_bit_number = _coerce_positive_bit(selection.get("bit_width"))
    if weight_bit_number is None:
        weight_bit_number = _extract_embedded_format_bit_number(
            _dashboard_text(row.get("weight_dt"))
        )
    if weight_bit_number is None:
        weight_bit_number = _coerce_positive_bit(row.get("w_bits"))
    return {
        "Series": series_name,
        "Bits": int(bit_number),
        "Input Bits": int(bit_number),
        "Weight Bits": (
            int(weight_bit_number) if weight_bit_number is not None else None
        ),
        "Candidates": candidate_label,
        "Full Evaluation": bool(full_evaluation),
        "Evaluation Scope Known": bool(scope_known),
        "Evaluation Scope": evaluation_scope,
        "Sweep Entry": bool(sweep_entry),
        "Sweep Direction": sweep_direction,
        "Weight Selection Mode": _dashboard_text(weight_selection.get("mode")),
        "Input Selection Mode": _dashboard_text(input_selection.get("mode")),
    }


def build_hybrid_accuracy_plot_df(df):
    """Build chart-ready accuracy rows for fixed and dynamic hybrid runs."""
    if df is None or df.empty:
        return pd.DataFrame()

    working_df = df.copy()
    if "w_bits" not in working_df.columns or "a_bits" not in working_df.columns:
        working_df = preprocess_runs_df(working_df)
    if working_df is None or working_df.empty or "acc1" not in working_df.columns:
        return pd.DataFrame()

    if "status" in working_df.columns:
        status_text = working_df["status"].fillna("").astype(str).str.upper()
        working_df = working_df[(status_text == "") | (status_text == "SUCCESS")]
    working_df["acc1_numeric"] = pd.to_numeric(
        working_df["acc1"], errors="coerce"
    )
    working_df = working_df.dropna(subset=["acc1_numeric"])

    rows = []
    for _, run_row in working_df.iterrows():
        metadata = _hybrid_accuracy_metadata(run_row)
        if metadata is None:
            continue
        weight_dt = _dashboard_text(run_row.get("weight_dt"), "unknown")
        activation_dt = _dashboard_text(
            run_row.get("activation_dt"), "unknown"
        )
        reference_accuracy = pd.to_numeric(
            run_row.get("ref_acc1_effective"), errors="coerce"
        )
        if pd.isna(reference_accuracy):
            reference_accuracy = pd.to_numeric(
                run_row.get("ref_acc1"), errors="coerce"
            )
        if metadata["Series"] == "Hybrid dynamic":
            input_variant = metadata["Candidates"] or activation_dt
            setup_label = f"W {weight_dt} · Dynamic [{input_variant}]"
        else:
            setup_label = f"W {weight_dt} · A {activation_dt}"
        rows.append({
            "Model": _dashboard_text(run_row.get("model_name"), "unknown"),
            **metadata,
            "Accuracy (%)": float(run_row["acc1_numeric"]),
            "Reference Accuracy (%)": reference_accuracy,
            "Setup Label": setup_label,
            "Weight DT": weight_dt,
            "Activation DT": activation_dt,
            "Variant": activation_dt,
            "Experiment Type": _dashboard_text(run_row.get("experiment_type")),
            "Run Date": _dashboard_text(run_row.get("run_date")),
            "Run Date Sort": pd.to_datetime(run_row.get("run_date"), errors="coerce"),
            "Run ID": run_row.get("id", None),
            "MSE": pd.to_numeric(run_row.get("mse"), errors="coerce"),
            "Certainty": pd.to_numeric(
                run_row.get("certainty"), errors="coerce"
            ),
        })

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return plot_df

    plot_df["Series"] = pd.Categorical(
        plot_df["Series"],
        categories=HYBRID_ACCURACY_SERIES_ORDER,
        ordered=True,
    )
    plot_df["Run ID Sort"] = pd.to_numeric(
        plot_df["Run ID"], errors="coerce"
    ).fillna(-1)
    plot_df = plot_df.sort_values(
        by=[
            "Model",
            "Bits",
            "Series",
            "Weight DT",
            "Activation DT",
            "Run Date Sort",
            "Run ID Sort",
        ],
        ascending=[True, True, True, True, True, False, False],
    )
    return plot_df.drop(columns=["Run Date Sort", "Run ID Sort"])


def build_hybrid_directional_plot_df(plot_df):
    """Build mirrored best-fixed hybrid sweeps and restore their shared point.

    The bidirectional runner logs the best-weight/best-input intersection once,
    under ``weight_fixed``.  That run is valid in both views, so this dashboard
    transform mirrors it into ``input_fixed`` when the combination is absent.
    """
    if plot_df is None or plot_df.empty:
        return pd.DataFrame()

    required = {
        "Model",
        "Sweep Direction",
        "Input Bits",
        "Weight Bits",
        "Weight DT",
        "Activation DT",
        "Accuracy (%)",
    }
    if not required.issubset(plot_df.columns):
        return pd.DataFrame()

    working = plot_df[
        plot_df["Sweep Direction"].isin(["weight_fixed", "input_fixed"])
    ].copy()
    if working.empty:
        return pd.DataFrame()

    # Older best-weight sweeps predate directional metadata.  Reuse only a
    # missing dynamic-input point whose weight exactly matches the explicit
    # per-width fixed weight; this repairs sparse historical sweeps without
    # blending unrelated legacy baselines into the new directional plots.
    legacy_dynamic = plot_df[
        plot_df["Sweep Direction"].eq("")
        & plot_df["Sweep Entry"].eq(True)
        & plot_df["Series"].astype(str).eq("Hybrid dynamic")
    ]
    compatibility_rows = []
    for (model, input_bits), explicit_group in working.groupby(
        ["Model", "Input Bits"], dropna=False
    ):
        weight_fixed = explicit_group[
            explicit_group["Sweep Direction"].eq("weight_fixed")
        ]
        if weight_fixed.empty:
            continue
        fixed_weights = set(weight_fixed["Weight DT"].dropna().astype(str))
        existing_pairs = set(zip(
            weight_fixed["Weight DT"].astype(str),
            weight_fixed["Activation DT"].astype(str),
        ))
        candidates = legacy_dynamic[
            legacy_dynamic["Model"].eq(model)
            & legacy_dynamic["Input Bits"].eq(input_bits)
            & legacy_dynamic["Weight DT"].astype(str).isin(fixed_weights)
        ]
        for _, row in candidates.iterrows():
            pair = (str(row["Weight DT"]), str(row["Activation DT"]))
            if pair in existing_pairs:
                continue
            compatible = row.copy()
            compatible["Sweep Direction"] = "weight_fixed"
            compatibility_rows.append(compatible)
            existing_pairs.add(pair)

    if compatibility_rows:
        working = pd.concat(
            [working, pd.DataFrame(compatibility_rows)], ignore_index=True
        )

    mirrored_rows = []
    group_columns = ["Model", "Input Bits"]
    for _, width_group in working.groupby(group_columns, dropna=False):
        input_fixed = width_group[
            width_group["Sweep Direction"].eq("input_fixed")
        ]
        weight_fixed = width_group[
            width_group["Sweep Direction"].eq("weight_fixed")
        ]
        if input_fixed.empty or weight_fixed.empty:
            continue

        fixed_inputs = set(input_fixed["Activation DT"].dropna().astype(str))
        existing_pairs = set(zip(
            input_fixed["Weight DT"].astype(str),
            input_fixed["Activation DT"].astype(str),
        ))
        for _, row in weight_fixed.iterrows():
            pair = (str(row["Weight DT"]), str(row["Activation DT"]))
            if pair in existing_pairs or pair[1] not in fixed_inputs:
                continue
            mirrored = row.copy()
            mirrored["Sweep Direction"] = "input_fixed"
            mirrored_rows.append(mirrored)
            existing_pairs.add(pair)

    if mirrored_rows:
        working = pd.concat(
            [working, pd.DataFrame(mirrored_rows)], ignore_index=True
        )

    rows = []
    for _, row in working.iterrows():
        direction = row["Sweep Direction"]
        if direction == "weight_fixed":
            axis_bits = row["Input Bits"]
            fixed_label = row["Weight DT"]
            candidate_label = row["Activation DT"]
            sweep_label = "Best weight → all inputs"
        else:
            axis_bits = row["Weight Bits"]
            fixed_label = row["Activation DT"]
            candidate_label = row["Weight DT"]
            sweep_label = "Best input → all weights"
        if pd.isna(axis_bits):
            continue
        rows.append({
            **row.to_dict(),
            "Axis Bits": int(axis_bits),
            "Fixed Label": str(fixed_label),
            "Candidate Label": str(candidate_label),
            "Sweep Label": sweep_label,
            "Winner": False,
        })

    directional_df = pd.DataFrame(rows)
    if directional_df.empty:
        return directional_df

    winner_indices = directional_df.groupby(
        ["Model", "Sweep Direction", "Axis Bits"], dropna=False
    )["Accuracy (%)"].idxmax()
    directional_df.loc[winner_indices, "Winner"] = True
    return directional_df.sort_values(
        ["Model", "Sweep Direction", "Axis Bits", "Accuracy (%)"],
        ascending=[True, True, True, False],
    )


def hybrid_best_dynamic_run_ids(plot_df):
    """Return the highest-accuracy dynamic run ID at each model and bit width."""
    if plot_df is None or plot_df.empty:
        return set()
    required = {"Model", "Bits", "Series", "Accuracy (%)", "Run ID"}
    if not required.issubset(plot_df.columns):
        return set()

    dynamic_rows = plot_df[
        plot_df["Series"].astype(str).eq("Hybrid dynamic")
        & plot_df["Run ID"].notna()
    ].copy()
    dynamic_rows["Accuracy Numeric"] = pd.to_numeric(
        dynamic_rows["Accuracy (%)"], errors="coerce"
    )
    dynamic_rows = dynamic_rows.dropna(
        subset=["Bits", "Accuracy Numeric", "Run ID"]
    )
    if dynamic_rows.empty:
        return set()

    best_indices = dynamic_rows.groupby(
        ["Model", "Bits"], observed=True, dropna=False
    )["Accuracy Numeric"].idxmax()
    return set(dynamic_rows.loc[best_indices, "Run ID"].tolist())


def filter_common_hybrid_width_rows(plot_df):
    """Keep hybrid rows whose weight and input widths are the same."""
    if plot_df is None or plot_df.empty:
        return pd.DataFrame()
    if not {"Weight Bits", "Input Bits"}.issubset(plot_df.columns):
        return pd.DataFrame()

    weight_bits = pd.to_numeric(plot_df["Weight Bits"], errors="coerce")
    input_bits = pd.to_numeric(plot_df["Input Bits"], errors="coerce")
    common_width = weight_bits.notna() & input_bits.notna() & weight_bits.eq(
        input_bits
    )
    return plot_df[common_width].copy()


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
        stays_on_chip = None

        if isinstance(value, dict):
            layer_type = str(value.get("type", "?"))
            stays_on_chip = value.get("stays_on_chip")
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

        layer_display = layer
        if stays_on_chip is True:
            layer_display = f"🟢 {layer}"
        elif stays_on_chip is False:
            layer_display = f"🔴 {layer}"

        layer_win_counts[dominant_format] = layer_win_counts.get(dominant_format, 0) + 1
        for fmt, cnt in counts.items():
            chunk_win_counts[fmt] = chunk_win_counts.get(fmt, 0) + int(cnt)
            layer_chunk_rows.append({
                "Layer": layer_display,
                "Layer Index": int(layer_idx),
                "Type": layer_type,
                "Format": fmt,
                "Chunk Wins": int(cnt),
            })

        layer_rows.append({
            "Layer": layer_display,
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
