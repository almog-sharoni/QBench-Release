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


def get_runs(limit):
    db = RunDatabase(db_path=DB_PATH)
    return db.get_runs(limit=limit)


def get_fm_runs(limit):
    if not os.path.exists(FM_DB_PATH):
        return pd.DataFrame()
    db = RunDatabase(db_path=FM_DB_PATH)
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

        if isinstance(value, dict):
            layer_type = str(value.get("type", "?"))
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

        layer_win_counts[dominant_format] = layer_win_counts.get(dominant_format, 0) + 1
        for fmt, cnt in counts.items():
            chunk_win_counts[fmt] = chunk_win_counts.get(fmt, 0) + int(cnt)
            layer_chunk_rows.append({
                "Layer": layer,
                "Layer Index": int(layer_idx),
                "Type": layer_type,
                "Format": fmt,
                "Chunk Wins": int(cnt),
            })

        layer_rows.append({
            "Layer": layer,
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


