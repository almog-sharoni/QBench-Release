# Chunk Layout Validator tab.
#
# Runs in the shared dashboard namespace (see dashboard.py): `st`, `RunDatabase`,
# `DB_PATH`, `pd`, `json` are already defined. Visualizes the canonical
# per-context chunker (chunk_tensor_by_context) the 128-PE hardware uses.
#
# Two views:
#   * Overview  — model-wide layers x format-share heatmap, built from the run's
#                 quant_map_json format_counts alone (no model build, instant).
#   * Detail    — per-context chunk grid for one layer, computed ANALYTICALLY
#                 (no tensor allocation) so it stays instant at any size.

with tab_chunk:
    import altair as _alt
    import pandas as _pd
    import json as _json

    # ---- analytic replica of chunk_tensor_by_context (validated cell-exact) --
    def _greedy(shape, cs):
        if not shape:
            return 1, 1, 1
        if len(shape) == 1:
            return 1, 1, int(shape[0])
        # Non-conv activations: greedily flatten trailing dimensions.
        if len(shape) >= 4:
            n = int(shape[0])
            h = int(shape[1])
            w = 1
            for d in shape[2:]:
                w *= int(d)
            return n, h, w
        n = int(shape[0]); dims = [int(d) for d in shape[1:]]; w = dims.pop()
        while dims and w * dims[-1] <= cs:
            w *= dims.pop()
        h = 1
        for d in dims:
            h *= d
        return n, h, w

    def _spatial_meta(shape, cs):
        if len(shape) < 4:
            return None
        n = int(shape[0])
        c = int(shape[1])
        row_width = int(shape[-1])
        rows_per_context = 1
        for d in shape[2:-1]:
            rows_per_context *= int(d)
        context_width = rows_per_context * row_width
        if context_width <= cs:
            contexts = n * c
            contexts_per_chunk = max(1, cs // max(context_width, 1))
            pad_contexts = (contexts_per_chunk - (contexts % contexts_per_chunk)) % contexts_per_chunk
            context_groups = (contexts + pad_contexts) // contexts_per_chunk
            group_width = contexts_per_chunk * context_width
            pad_width = cs - group_width
            return {
                "kind": "packed_spatial_contexts",
                "n_ctx": context_groups,
                "contexts": contexts,
                "context_width": context_width,
                "contexts_per_chunk": contexts_per_chunk,
                "pad_contexts": pad_contexts,
                "context_groups": context_groups,
                "group_width": group_width,
                "pad_width": pad_width,
            }
        rows_per_chunk = max(1, cs // max(row_width, 1))
        rows_per_chunk = min(rows_per_chunk, max(1, rows_per_context))
        pad_rows = (rows_per_chunk - (rows_per_context % rows_per_chunk)) % rows_per_chunk
        rows_padded = rows_per_context + pad_rows
        row_groups = rows_padded // rows_per_chunk
        group_width = rows_per_chunk * row_width
        pad_width = (cs - (group_width % cs)) % cs
        chunks_per_group = (group_width + pad_width) // cs
        return {
            "kind": "spatial_rows",
            "n_ctx": n * c,
            "rows_per_context": rows_per_context,
            "row_width": row_width,
            "rows_per_chunk": rows_per_chunk,
            "pad_rows": pad_rows,
            "row_groups": row_groups,
            "group_width": group_width,
            "pad_width": pad_width,
            "chunks_per_group": chunks_per_group,
        }

    def _factor(h):
        f = 1 if h > 64 else 2 if h > 32 else 4 if h > 16 else 8 if h > 8 else 16
        return min(f, max(1, int(h)))

    @st.cache_data(show_spinner=False)
    def _chunk_structure(shape, cs=128):
        """Return chunk layout for `shape` without allocating any tensor.

        n_ctx, n_chunk and the per-(merged-row, chunk) real-element counts match
        chunk_tensor_by_context exactly. real counts depend only on the merged-row
        index j (0..rows_per_n-1), so we store one row of counts per j and tile.
        """
        shape = tuple(int(s) for s in shape)
        meta = _spatial_meta(shape, cs)
        if meta is not None:
            if meta["kind"] == "packed_spatial_contexts":
                perj = []
                for j in range(meta["context_groups"]):
                    real_contexts = max(0, min(
                        meta["contexts"] - j * meta["contexts_per_chunk"],
                        meta["contexts_per_chunk"],
                    ))
                    perj.append([real_contexts * meta["context_width"]])
                return _finalize_structure(meta["n_ctx"], 1, cs, meta["context_groups"], perj)

            n_ctx = meta["n_ctx"]
            n_chunk = meta["row_groups"] * meta["chunks_per_group"]
            row = []
            for j in range(meta["row_groups"]):
                real_rows = max(0, min(
                    meta["rows_per_context"] - j * meta["rows_per_chunk"],
                    meta["rows_per_chunk"],
                ))
                real_elems = real_rows * meta["row_width"]
                for k in range(meta["chunks_per_group"]):
                    row.append(max(0, min(real_elems - k * cs, cs)))
            return _finalize_structure(n_ctx, n_chunk, cs, 1, [row])

        n, h_local, w = _greedy(shape, cs)
        factor = _factor(h_local)
        h_padded = ((h_local + factor - 1) // factor) * factor if factor > 1 else h_local
        merged_w = factor * w if factor > 1 else w
        rows_per_n = max(1, h_padded // factor)
        pad_len = (cs - (merged_w % cs)) % cs
        n_chunk = (merged_w + pad_len) // cs
        n_ctx = n * rows_per_n
        perj = []
        for j in range(rows_per_n):
            real_sub = max(0, min(h_local - j * factor, factor))
            real_elems = real_sub * w
            perj.append([max(0, min(real_elems - k * cs, cs)) for k in range(n_chunk)])
        return _finalize_structure(n_ctx, n_chunk, cs, rows_per_n, perj)

    def _finalize_structure(n_ctx, n_chunk, cs, rows_per_n, perj):
        tile = max(1, n_ctx // max(rows_per_n, 1))
        total_chunks = n_ctx * n_chunk
        real_chunks = sum(1 for row in perj for v in row if v > 0) * tile
        real_slots = sum(v for row in perj for v in row) * tile
        return {
            "n_ctx": n_ctx, "n_chunk": n_chunk, "cs": cs, "rows_per_n": rows_per_n,
            "perj": perj, "total_chunks": total_chunks,
            "real_chunks": real_chunks, "pad_chunks": total_chunks - real_chunks,
            "real_slots": real_slots, "total_slots": total_chunks * cs,
        }

    @st.cache_data(show_spinner=False)
    def _chunk_structure_weight(shape, cs=128):
        """Analytic weight chunk layout (matches chunk_weight_by_context)."""
        shape = tuple(int(s) for s in shape)
        c_out = shape[0] if shape else 1
        kernel = 1
        for d in shape[2:]:
            kernel *= d
        numel = 1
        for d in shape:
            numel *= d
        fan_in = numel // c_out if c_out else numel
        if fan_in <= cs:                                  # pack whole channels
            k = max(1, cs // fan_in)
            n_groups = -(-c_out // k)
            perj = [[min(k, c_out - g * k) * fan_in] for g in range(n_groups)]
            return _finalize_structure(n_groups, 1, cs, n_groups, perj)
        b = max(1, cs // kernel)                          # pack whole f*f blocks
        c_in = fan_in // kernel
        n_chunk = -(-c_in // b)
        perj = [[min(b, c_in - j * b) * kernel for j in range(n_chunk)]]
        return _finalize_structure(c_out, n_chunk, cs, 1, perj)

    @st.cache_data(show_spinner="Introspecting model layer shapes…")
    def _chunk_layer_shapes(model_name: str):
        import torchvision.models as _tvm
        import torch.nn as _nn
        ctor = getattr(_tvm, model_name, None)
        if ctor is None:
            return {}
        try:
            model = ctor(weights=None)
        except Exception:
            try:
                model = ctor()
            except Exception:
                return {}
        shapes = {}
        for name, mod in model.named_modules():
            if isinstance(mod, (_nn.Conv2d, _nn.Linear)) and getattr(mod, "weight", None) is not None:
                if mod.weight.dim() >= 2:
                    shapes[name] = tuple(int(s) for s in mod.weight.shape)
        return shapes

    @st.cache_data(show_spinner=False)
    def _chunk_all_runs(db_path: str):
        try:
            return RunDatabase(db_path).get_runs()
        except Exception:
            return _pd.DataFrame()

    def _runs_with_map(rdf, col):
        """Runs carrying a usable map in column `col` (quant_map_json | input_map_json)."""
        if rdf is None or rdf.empty or col not in rdf.columns:
            return _pd.DataFrame()
        sub = rdf[rdf[col].notna() & (rdf[col].astype(str).str.len() > 2)]
        return sub.reset_index(drop=True)

    @st.cache_data(show_spinner="Tracing input shapes (forward pass)…")
    def _chunk_input_shapes(model_name: str, batch: int = 1):
        """Per-layer INPUT tensor shapes via a dummy forward (activations target)."""
        import torch as _t
        import torch.nn as _nn
        import torchvision.models as _tvm
        ctor = getattr(_tvm, model_name, None)
        if ctor is None:
            return {}
        try:
            model = ctor(weights=None).eval()
        except Exception:
            try:
                model = ctor().eval()
            except Exception:
                return {}
        shapes, hooks = {}, []

        def _mk(nm):
            def _h(m, inp, out):
                if inp and isinstance(inp[0], _t.Tensor):
                    shapes[nm] = tuple(int(s) for s in inp[0].shape)
            return _h
        for nm, mod in model.named_modules():
            if isinstance(mod, (_nn.Conv2d, _nn.Linear)) and getattr(mod, "weight", None) is not None:
                hooks.append(mod.register_forward_hook(_mk(nm)))
        try:
            from runspace.src.utils.model_input_utils import resolve_model_input_size
            in_shape = resolve_model_input_size(model, batch_size=batch)
        except Exception:
            in_shape = (batch, 3, 224, 224)
        try:
            with _t.no_grad():
                try:
                    model(_t.randn(*in_shape))
                except Exception:
                    model((_t.randn(*in_shape), None))
        except Exception:
            pass
        for h in hooks:
            h.remove()
        return shapes

    def _entry_formats(spec):
        """(per_chunk_list_or_None, counts_dict, total_chunks_or_None) for a map entry."""
        if isinstance(spec, list):
            lst = [str(f) for f in spec]
            counts = {}
            for f in lst:
                counts[f] = counts.get(f, 0) + 1
            return lst, counts, len(lst)
        if isinstance(spec, dict):
            f = spec.get("format")
            lst = [str(x) for x in f] if isinstance(f, list) else None
            counts = spec.get("format_counts")
            if not isinstance(counts, dict) and lst is not None:
                counts = {}
                for x in lst:
                    counts[x] = counts.get(x, 0) + 1
            counts = {str(k): int(v) for k, v in (counts or {}).items()}
            tc = spec.get("total_chunks")
            return lst, counts, (int(tc) if tc is not None else (len(lst) if lst else None))
        if isinstance(spec, str):
            return None, {spec: 1}, 1
        return None, {}, None

    _PALETTE = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#eeca3b",
                "#b279a2", "#ff9da6", "#9d755d", "#bab0ac", "#1f77b4", "#ff7f0e",
                "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    def _fmt_bits(f):
        try:
            return int(str(f).split("fp")[1].split("_")[0])
        except Exception:
            return 99

    def _color_scale(all_formats, with_padding=True):
        fmts = sorted(set(all_formats), key=lambda f: (_fmt_bits(f), str(f)))
        domain = list(fmts) + (["padding"] if with_padding else [])
        rng = [_PALETTE[i % len(_PALETTE)] for i in range(len(fmts))] + (["#e8e8e8"] if with_padding else [])
        return _alt.Scale(domain=domain, range=rng)

    def _render_chunk_grid(shape, cs, overlay_formats, run_formats=None,
                           ctx_label="context (output channel)", structure_fn=None):
        """Per-context chunk grid for one tensor, computed analytically (no alloc)."""
        info = (structure_fn or _chunk_structure)(tuple(shape), cs)
        n_ctx, n_chunk, cs = info["n_ctx"], info["n_chunk"], info["cs"]
        rpn, perj = info["rows_per_n"], info["perj"]
        total, pad_chunks = info["total_chunks"], info["pad_chunks"]
        fill_pct = 100.0 * info["real_slots"] / max(info["total_slots"], 1)
        nonempty = [v for row in perj for v in row if v > 0]
        pe_util = 100.0 * (sum(nonempty) / (len(nonempty) * cs)) if nonempty else 0.0
        numel = 1
        for s in shape:
            numel *= int(s)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Contexts", f"{n_ctx:,}")
        m2.metric("Chunks / context", f"{n_chunk:,}")
        m3.metric("Total chunks", f"{total:,}")
        m4.metric("Padding chunks", f"{pad_chunks:,}",
                  delta=f"{100.0*pad_chunks/max(total,1):.0f}%", delta_color="inverse")
        m5.metric("PE utilization", f"{pe_util:.0f}%",
                  help="Mean real fill of non-empty 128-chunks. Low ⇒ this layer tiles poorly.")

        if overlay_formats is not None and len(overlay_formats) != total:
            flat = -(-numel // cs)
            st.warning(
                f"Run map has **{len(overlay_formats)}** entries but the per-context layout has "
                f"**{total}** chunks (= {n_ctx}×{n_chunk}). Mismatch ⇒ almost certainly a pre-fix "
                f"flat-chunked run (flat blocks = ⌈{numel}/{cs}⌉ = {flat}). Overlay disabled."
            )
            overlay_formats = None

        opts = (["Format"] if overlay_formats is not None else []) + ["Fill ratio", "Real / padding"]
        color_by = st.radio("Color by", opts, horizontal=True, key="chunk_grid_colorby")

        max_cells = 6000
        max_ctx = max(1, min(n_ctx, max(1, max_cells // max(n_chunk, 1))))
        start = 0
        if n_ctx > max_ctx:
            start = st.slider(f"Context window (showing {max_ctx} of {n_ctx})",
                              0, n_ctx - max_ctx, 0, key="chunk_grid_window")
        end = min(n_ctx, start + max_ctx)

        rows = []
        for ctx in range(start, end):
            j = ctx % rpn
            base = ctx * n_chunk
            for k in range(n_chunk):
                r = perj[j][k]
                is_pad = r == 0
                fmt = "padding" if is_pad else (overlay_formats[base + k] if overlay_formats is not None else "real")
                rows.append({"context": ctx, "chunk": k, "real": r, "pad": cs - r,
                             "fill": r / cs, "format": fmt, "kind": "padding" if is_pad else "real"})
        ddf = _pd.DataFrame(rows)
        tip = [_alt.Tooltip("context:Q", title="ctx (out ch)"), _alt.Tooltip("chunk:Q", title="chunk"),
               _alt.Tooltip("real:Q", title="real"), _alt.Tooltip("pad:Q", title="pad"),
               _alt.Tooltip("fill:Q", title="fill", format=".0%"), _alt.Tooltip("format:N", title="fmt")]

        if color_by == "Format":
            fmts = set(run_formats or []) | {f for f in ddf["format"] if f != "padding"}
            color = _alt.Color("format:N", scale=_color_scale(fmts, with_padding=True),
                               legend=_alt.Legend(title="format", orient="top"))
        elif color_by == "Fill ratio":
            color = _alt.Color("fill:Q", scale=_alt.Scale(scheme="viridis", domain=[0, 1]),
                               legend=_alt.Legend(title="fill", format=".0%"))
        else:
            color = _alt.Color("kind:N", scale=_alt.Scale(domain=["real", "padding"], range=["#4c78a8", "#e8e8e8"]),
                               legend=_alt.Legend(title=""))

        cell = max(5, min(24, int(880 / max(n_chunk, 1))))
        dense_chunks = n_chunk > 240
        stroke_width = 0 if dense_chunks else 0.3
        x_axis = _alt.Axis(
            labels=not dense_chunks,
            ticks=not dense_chunks,
            title=f"chunk idx (each = {cs})",
        )
        chart_height = max(120, min(640, cell * (end - start) + 30))
        chart = (
            _alt.Chart(ddf).mark_rect(stroke="#ffffff", strokeWidth=stroke_width)
            .encode(x=_alt.X("chunk:O", axis=x_axis),
                    y=_alt.Y("context:O", sort=list(range(start, end)), title=ctx_label),
                    color=color, tooltip=tip)
            .properties(height=chart_height, width="container")
            .interactive()
        )
        st.altair_chart(chart)

    # ========================================================================
    st.subheader("🧩 Chunk Layout Validator")
    st.caption(
        "The 128-PE hardware MACs one context at a time: one context = one output "
        "channel, its fan-in tiled into 128-element chunks that never cross contexts."
    )

    _db_path = globals().get("DB_PATH") or RunDatabase()._default_db_path()

    _hdr, _btn = st.columns([4, 1])
    _hdr.caption(f"Runs read from `{_db_path}` (cached). Click reload after re-running experiments.")
    if _btn.button("🔄 Reload from DB", key="chunk_reload"):
        _chunk_all_runs.clear()
        try:
            get_runs.clear()  # also refresh the shared dashboard cache if present
        except Exception:
            pass
        st.rerun()
    all_runs = _chunk_all_runs(_db_path)

    mode = st.radio("Mode", ["From a run (formats)", "Explore the chunker (shape only)"],
                    horizontal=True, key="chunk_mode")

    # ------------------------------------------------------------------ run mode
    if mode == "From a run (formats)":
        target = st.radio("Target", ["Weights", "Inputs (activations)"], horizontal=True, key="chunk_target")
        is_weights = target == "Weights"
        map_col = "quant_map_json" if is_weights else "input_map_json"
        ctx_label = "context (output channel)" if is_weights else "context (activation row)"
        runs_df = _runs_with_map(all_runs, map_col)
        if not is_weights and not runs_df.empty and "activation_dt" in runs_df.columns:
            runs_df = runs_df[runs_df["activation_dt"].fillna("fp32").astype(str).str.lower() != "fp32"]
            runs_df = runs_df.reset_index(drop=True)

        if runs_df.empty:
            st.warning(f"No runs carry a `{map_col}` in this DB. Try the other target or *Explore the chunker* mode.")
        else:
            def _run_label(i):
                r = runs_df.iloc[i]
                dt = r.get("weight_dt") if is_weights else r.get("activation_dt")
                return f"[{r.get('id','?')}] {r.get('model_name','?')} · {dt} · {r.get('experiment_type','?')}"
            sel_i = st.selectbox(
                "Run",
                list(range(len(runs_df))),
                format_func=_run_label,
                key=f"chunk_run_{'weights' if is_weights else 'inputs'}",
            )
            row = runs_df.iloc[sel_i]
            run_model = str(row.get("model_name") or "")
            try:
                qmap = _json.loads(row[map_col]) if isinstance(row[map_col], str) else row[map_col]
            except Exception:
                qmap = None

            if not isinstance(qmap, dict) or not qmap:
                st.error(f"Selected run has no usable `{map_col}`.")
            else:
                # ---- model-wide overview (instant: counts only) ---------------
                ov_rows = []
                for layer, spec in qmap.items():
                    _lst, counts, tc = _entry_formats(spec)
                    tot = sum(counts.values()) or 1
                    for fmt, cnt in counts.items():
                        ov_rows.append({"layer": layer, "format": fmt, "chunks": cnt,
                                        "share": cnt / tot})
                ov = _pd.DataFrame(ov_rows)
                all_fmts = sorted(ov["format"].unique(), key=lambda f: (_fmt_bits(f), f))
                scale = _color_scale(all_fmts, with_padding=False)
                layer_order = list(qmap.keys())

                st.markdown(f"**Model-wide {target.split()[0].lower()} format selection** — `{run_model}`, {len(layer_order)} layers")
                show_share = st.toggle("Normalize (share of chunks)", value=True, key="chunk_ov_norm")
                xfield = _alt.X("share:Q", stack="normalize", title="share of chunks", axis=_alt.Axis(format="%")) \
                    if show_share else _alt.X("chunks:Q", stack="zero", title="chunks")
                overview = (
                    _alt.Chart(ov)
                    .mark_bar()
                    .encode(
                        y=_alt.Y("layer:N", sort=layer_order, title=None,
                                 axis=_alt.Axis(labelLimit=300, labelFontSize=10)),
                        x=xfield,
                        color=_alt.Color("format:N", scale=scale, legend=_alt.Legend(title="format", orient="top")),
                        order=_alt.Order("format:N"),
                        tooltip=["layer:N", "format:N", "chunks:Q", _alt.Tooltip("share:Q", format=".1%")],
                    )
                    .properties(height=min(1400, 16 * len(layer_order) + 30), width="container")
                )
                st.altair_chart(overview)

                st.divider()
                # ---- drill-down detail grid -----------------------------------
                grid_what = "weight" if is_weights else "input"
                st.markdown(f"**Per-context chunk grid** — drill into one layer's {grid_what} tensor")
                shapes = (_chunk_layer_shapes(run_model) if is_weights else _chunk_input_shapes(run_model)) if run_model else {}
                drill_layers = [l for l in layer_order if (not shapes) or (l in shapes)]
                dl1, dl2 = st.columns([3, 1])
                layer_name = dl1.selectbox("Layer", drill_layers,
                                           format_func=lambda n: f"{n}  {tuple(shapes[n])}" if n in shapes else n,
                                           key=f"chunk_detail_layer_{'weights' if is_weights else 'inputs'}")
                cs = int(dl2.number_input(
                    "Chunk size", 8, 1024, 128, 8,
                    key=f"chunk_cs_run_{'weights' if is_weights else 'inputs'}",
                ))
                shape = shapes.get(layer_name)
                overlay_formats, _counts, _tc = _entry_formats(qmap.get(layer_name))
                if not is_weights and overlay_formats is None:
                    st.caption(
                        "Input formats are chosen dynamically per sample, so only aggregate "
                        "counts are stored — the grid shows the activation's chunk *layout* "
                        "(tiling / padding / PE utilization); color by fill or padding."
                    )
                if shape is None:
                    st.warning(
                        f"No traced {'weight' if is_weights else 'input'} shape for `{layer_name}`."
                        + ("" if is_weights else " (Forward trace may have skipped this layer.)"))
                else:
                    _render_chunk_grid(shape, cs, overlay_formats, run_formats=all_fmts, ctx_label=ctx_label,
                                       structure_fn=(_chunk_structure_weight if is_weights else _chunk_structure))

    # -------------------------------------------------------------- explore mode
    else:
        src = st.radio("Tensor source", ["torchvision layer", "Manual shape"], horizontal=True, key="chunk_src2")
        interp = st.radio("Chunk as", ["Weight", "Input (activation)"], horizontal=True, key="chunk_interp")
        c1, c2 = st.columns([3, 1])
        cs = int(c2.number_input("Chunk size", 8, 1024, 128, 8, key="chunk_cs_expl"))
        shape = None
        if src == "torchvision layer":
            common = ["resnet18", "resnet50", "efficientnet_b0", "mobilenet_v3_large",
                      "vgg16", "vit_b_16", "densenet161", "alexnet"]
            db_models = sorted(all_runs["model_name"].dropna().unique().tolist()) if not all_runs.empty else []
            mname = c1.selectbox("Model", sorted(set(common) | set(db_models)), key="chunk_expl_model")
            shapes = _chunk_layer_shapes(mname)
            if shapes:
                ln = st.selectbox("Layer", list(shapes.keys()),
                                  format_func=lambda n: f"{n}  {tuple(shapes[n])}", key="chunk_expl_layer")
                shape = shapes.get(ln)
            else:
                st.warning(f"Could not introspect `{mname}`.")
        else:
            txt = c1.text_input("Weight shape", value="64,3,7,7", key="chunk_expl_shape")
            try:
                shape = tuple(int(p) for p in txt.replace("(", "").replace(")", "").replace("x", ",").split(",") if p.strip())
            except Exception:
                st.error("Parse error — use e.g. `1000,1280` or `64,3,7,7`.")
        if shape:
            _w = interp == "Weight"
            _render_chunk_grid(shape, cs, None,
                               ctx_label="context (output channel)" if _w else "context (activation row)",
                               structure_fn=_chunk_structure_weight if _w else _chunk_structure)
