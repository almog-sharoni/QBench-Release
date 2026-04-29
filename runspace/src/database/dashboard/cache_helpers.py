# ── Cache Simulation Tab ──────────────────────────────────────────────────────
def _fmt_e(n):
    """Format element count for display."""
    try:
        n = int(n)
    except Exception:
        return str(n)
    if n >= 1_000_000:
        return f"{n/1_000_000:.3f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def _build_bank_states(layers, num_banks, bank_size, rule_meta=None):
    """
    Pre-compute the cache bank state *during* each layer's execution.

    rule_meta: optional dict of {rule_name: {'xin_from_cache': bool, 'permanents': str}}
               built from the simulation's rules_json.  When provided, xin visibility
               and weight-bank display are derived from rule properties instead of
               hardcoded rule names, so the viz stays correct as rules change.
    """
    import math

    # Build fast per-rule lookups.
    # xin_full:        True  → xin fully resident in cache during execution
    # wt_shows:        True  → weight banks are explicitly held during execution
    # pipeline_banks:  int   → extra boundary banks for xin-on-xout overlap rules (xin not shown separately)
    _xin_full       = {}
    _wt_shows       = {}
    _pipeline_banks = {}
    if rule_meta:
        for name, meta in rule_meta.items():
            _xin_full[name]       = bool(meta.get('xin_from_cache', False))
            _wt_shows[name]       = 'weight' in meta.get('permanents', '').lower()
            _pipeline_banks[name] = int(meta.get('pipeline_banks', 0))
    else:
        # Legacy fallback: covers both old (r1_/r2_/r3_) and new short names.
        for name in ('r1_global_fit', 'r2_residual', 'r2_pool',
                     'global_fit', 'residual', 'pool'):
            _xin_full[name] = True
            _wt_shows[name] = False
        for name in ('r2_conv_output_dominated', 'r2_conv_input_dominated',
                     'conv_output_dominated', 'conv_input_dominated',
                     'linear_stream_xout'):
            _xin_full[name]       = True
            _wt_shows[name]       = True
            _pipeline_banks[name] = 1
        for name in ('r2_stream_xin_keep_xout', 'stream_xin_keep_xout',
                     'r3_weights_plus_4banks', 'fallback'):
            _xin_full[name] = False
            _wt_shows[name] = True

    states = []
    for layer in layers:
        rule = layer.get('rule', '')
        stay = bool(layer.get('stay_on_chip', False))
        oe   = int(layer.get('output_elems', 0) or 0)
        we   = int(layer.get('weight_elems', 0) or 0)
        ie   = int(layer.get('input_elems',  0) or 0)

        ob = math.ceil(oe / bank_size) if oe > 0 else 0
        wb = math.ceil(we / bank_size) if we > 0 else 0
        ib = math.ceil(ie / bank_size) if ie > 0 else 0

        ob = min(ob, num_banks)
        wb = min(wb, num_banks)
        ib = min(ib, num_banks)

        xin_full      = _xin_full.get(rule, False)
        wt_shows      = _wt_shows.get(rule, False)
        pipeline_b    = _pipeline_banks.get(rule, 0)

        if pipeline_b > 0:
            # xin is written onto the dominant tensor's space; only pipeline boundary banks shown
            xin_b    = 0
            stream_b = pipeline_b
        elif xin_full:
            xin_b    = ib
            stream_b = 0
        else:
            xin_b    = 0
            stream_b = 2       # 2-bank streaming buffer for xin from external

        xr_b = 0
        layer_type = layer.get('type', '')
        if stay:
            if layer_type == 'Residual':
                # xin (skip) fully resident; xout + x_r each use 2 streaming banks
                xout_b, xr_b, wt_b = 2, 2, 0
            else:
                xout_b = ob
                if wt_shows:
                    wt_b = wb          # weights permanently resident (e.g. conv_output_dominated)
                elif we > 0:
                    wt_b = min(2, wb)  # 2 streaming banks for weight tiles (e.g. global_fit on Conv2d)
                else:
                    wt_b = 0           # no weights (pool)
        else:
            # off-chip: xout streamed out; weights may stay resident for streaming
            xout_b   = 0
            wt_b     = min(wb, num_banks - stream_b - 2) if wt_shows else 0
            stream_b = min(stream_b + 2, num_banks - wt_b)  # xin-stream + xout-stream

        used   = xin_b + xout_b + xr_b + wt_b + stream_b
        free_b = max(0, num_banks - used)

        # Build per-bank list
        banks = []
        for _ in range(xin_b):
            banks.append({'type': 'xin', 'label': 'xin'})
        for _ in range(xout_b):
            banks.append({'type': 'xout', 'label': 'xout'})
        for _ in range(xr_b):
            banks.append({'type': 'xr', 'label': 'xr'})
        for _ in range(wt_b):
            banks.append({'type': 'weights', 'label': 'W'})
        for _ in range(stream_b):
            banks.append({'type': 'stream', 'label': '~'})
        for _ in range(free_b):
            banks.append({'type': 'free', 'label': ''})
        while len(banks) < num_banks:
            banks.append({'type': 'free', 'label': ''})
        banks = banks[:num_banks]

        states.append({
            'name':     layer.get('name', ''),
            'type':     layer.get('type', ''),
            'rule':     rule,
            'reason':   layer.get('reason', ''),
            'stay':     stay,
            'xin_b':    xin_b,
            'xout_b':   xout_b,
            'xr_b':     xr_b,
            'wt_b':     wt_b,
            'stream_b': stream_b,
            'free_b':   free_b,
            'output':   _fmt_e(oe),
            'weights':  _fmt_e(we),
            'input':    _fmt_e(ie),
            'banks':    banks,
        })
    return states


def _render_bank_viz_html(states, num_banks, bank_size, cache_elements):
    """Return a self-contained HTML string for the interactive bank viewer."""
    states_json = json.dumps(states)
    bank_size_fmt = _fmt_e(bank_size)
    cache_fmt = _fmt_e(cache_elements)
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
          background: #f8fafc; color: #0f172a; padding: 14px; }}
  h3 {{ font-size: 13px; font-weight: 700; color: #475569; letter-spacing:.06em;
        text-transform: uppercase; margin-bottom: 10px; }}
  #controls {{ display:flex; align-items:center; gap:12px; margin-bottom:14px; flex-wrap:wrap; }}
  #layer-slider {{ flex:1; min-width:200px; accent-color:#0f766e; }}
  #layer-counter {{ font-size:12px; color:#64748b; white-space:nowrap; }}
  #nav-btns {{ display:flex; gap:6px; }}
  .nav-btn {{ cursor:pointer; background:#fff; border:1px solid #cbd5e1; border-radius:7px;
              color:#334155; padding:4px 10px; font-size:12px; font-weight:600;
              transition:background .1s; }}
  .nav-btn:hover {{ background:#f1f5f9; }}

  #info-panel {{ background:#fff; border:1px solid #e2e8f0; border-radius:12px;
                 padding:12px 14px; margin-bottom:14px; }}
  .info-row {{ display:flex; gap:20px; flex-wrap:wrap; margin-bottom:6px; }}
  .info-label {{ font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase;
                 letter-spacing:.05em; }}
  .info-val {{ font-size:13px; font-weight:600; color:#0f172a; }}
  .rule-pill {{ display:inline-block; background:#eff6ff; border:1px solid #bfdbfe;
                border-radius:999px; padding:2px 10px; font-size:11px; font-weight:700;
                color:#1d4ed8; }}
  .oncehip-pill {{ background:#f0fdf4; border-color:#86efac; color:#15803d; }}
  .offchip-pill  {{ background:#fef2f2; border-color:#fca5a5; color:#dc2626; }}

  #bank-container {{ display:flex; gap:4px; flex-wrap:nowrap; margin-bottom:8px; }}
  .bank {{ flex:1; min-width:0; border-radius:6px; display:flex; flex-direction:column;
           align-items:center; justify-content:center; padding:8px 2px; position:relative;
           transition: background .25s, border-color .25s; border:1.5px solid transparent;
           cursor:default; }}
  .bank:hover {{ border-color: #94a3b8 !important; }}
  .bank-num {{ font-size:9px; color:rgba(0,0,0,.35); font-weight:600; position:absolute;
               bottom:3px; }}
  .bank-lbl {{ font-size:10px; font-weight:700; color:rgba(0,0,0,.55); }}
  .bank.xin     {{ background:#fed7aa; }}
  .bank.xout    {{ background:#dcfce7; }}
  .bank.xr      {{ background:#f3e8ff; }}
  .bank.weights {{ background:#dbeafe; }}
  .bank.stream  {{ background:#fef9c3; }}
  .bank.free    {{ background:#f1f5f9; }}

  #legend {{ display:flex; gap:14px; flex-wrap:wrap; margin-bottom:12px; }}
  .leg-item {{ display:flex; align-items:center; gap:5px; font-size:11px; font-weight:600; color:#475569; }}
  .leg-dot {{ width:12px; height:12px; border-radius:3px; }}
  .leg-xin     {{ background:#fed7aa; border:1px solid #fb923c; }}
  .leg-xout    {{ background:#dcfce7; border:1px solid #86efac; }}
  .leg-xr      {{ background:#f3e8ff; border:1px solid #c084fc; }}
  .leg-weights {{ background:#dbeafe; border:1px solid #93c5fd; }}
  .leg-stream  {{ background:#fef9c3; border:1px solid #fde047; }}
  .leg-free    {{ background:#f1f5f9; border:1px solid #cbd5e1; }}

  #size-bar {{ display:flex; height:20px; border-radius:6px; overflow:hidden; margin-bottom:12px;
               border:1px solid #e2e8f0; }}
  .sz-seg {{ height:100%; display:flex; align-items:center; justify-content:center;
             font-size:10px; font-weight:700; color:rgba(0,0,0,.5); transition:width .3s; }}
  .sz-xin     {{ background:#fdba74; }}
  .sz-xout    {{ background:#bbf7d0; }}
  .sz-xr      {{ background:#e9d5ff; }}
  .sz-weights {{ background:#bfdbfe; }}
  .sz-stream  {{ background:#fef08a; }}
  .sz-free    {{ background:#f1f5f9; }}

  #stats {{ display:flex; gap:12px; flex-wrap:wrap; }}
  .stat-box {{ background:#fff; border:1px solid #e2e8f0; border-radius:10px;
               padding:8px 14px; min-width:80px; text-align:center; }}
  .stat-val {{ font-size:18px; font-weight:700; color:#0f172a; }}
  .stat-lbl {{ font-size:10px; color:#64748b; font-weight:600; text-transform:uppercase; }}
  .stat-box.orange {{ border-color:#fb923c; background:#fff7ed; }}
  .stat-box.green  {{ border-color:#86efac; background:#f0fdf4; }}
  .stat-box.blue   {{ border-color:#93c5fd; background:#eff6ff; }}
  .stat-box.yellow {{ border-color:#fde047; background:#fefce8; }}
  .stat-box.gray   {{ border-color:#cbd5e1; background:#f8fafc; }}
  .stat-box.red    {{ border-color:#fca5a5; background:#fef2f2; }}
</style>
</head>
<body>
<h3>Memory Banks — {num_banks} banks × {bank_size_fmt} elem = {cache_fmt} elem total</h3>

<div id="controls">
  <div id="nav-btns">
    <button class="nav-btn" onclick="step(-1)">◀ Prev</button>
    <button class="nav-btn" onclick="step(1)">Next ▶</button>
  </div>
  <input type="range" id="layer-slider" min="0" max="0" value="0" oninput="setLayer(+this.value)">
  <span id="layer-counter"></span>
</div>

<div id="info-panel">
  <div class="info-row">
    <div><div class="info-label">Layer</div><div class="info-val" id="i-name">—</div></div>
    <div><div class="info-label">Type</div><div class="info-val" id="i-type">—</div></div>
    <div><div class="info-label">xin</div><div class="info-val" id="i-input">—</div></div>
    <div><div class="info-label">xout</div><div class="info-val" id="i-output">—</div></div>
    <div><div class="info-label">Weights</div><div class="info-val" id="i-weights">—</div></div>
  </div>
  <div class="info-row">
    <div><div class="info-label">Rule</div><div class="info-val"><span class="rule-pill" id="i-rule">—</span></div></div>
    <div><div class="info-label">Decision</div><div class="info-val"><span class="rule-pill" id="i-stay">—</span></div></div>
    <div style="flex:1"><div class="info-label">Reason</div><div class="info-val" id="i-reason" style="font-size:12px;color:#475569">—</div></div>
  </div>
</div>

<div id="legend">
  <div class="leg-item"><div class="leg-dot leg-xin"></div> xin (in cache)</div>
  <div class="leg-item"><div class="leg-dot leg-xout"></div> xout (on-chip output)</div>
  <div class="leg-item"><div class="leg-dot leg-xr"></div> x_r (residual stream)</div>
  <div class="leg-item"><div class="leg-dot leg-weights"></div> Weights</div>
  <div class="leg-item"><div class="leg-dot leg-stream"></div> Streaming buffer (~)</div>
  <div class="leg-item"><div class="leg-dot leg-free"></div> Free</div>
</div>

<div id="size-bar">
  <div class="sz-seg sz-xin"     id="sz-xin"     style="width:0%"></div>
  <div class="sz-seg sz-xout"    id="sz-xout"    style="width:0%"></div>
  <div class="sz-seg sz-xr"      id="sz-xr"      style="width:0%"></div>
  <div class="sz-seg sz-weights" id="sz-weights" style="width:0%"></div>
  <div class="sz-seg sz-stream"  id="sz-stream"  style="width:0%"></div>
  <div class="sz-seg sz-free"    id="sz-free"    style="width:100%">free</div>
</div>

<div id="bank-container"></div>

<div id="stats">
  <div class="stat-box orange"><div class="stat-val" id="st-xin">0</div><div class="stat-lbl">xin banks</div></div>
  <div class="stat-box green" ><div class="stat-val" id="st-xout">0</div><div class="stat-lbl">xout banks</div></div>
  <div class="stat-box" id="xr-box" style="border-color:#c084fc;background:#faf5ff;display:none">
    <div class="stat-val" id="st-xr">0</div><div class="stat-lbl">x_r banks</div>
  </div>
  <div class="stat-box blue"  ><div class="stat-val" id="st-wt">0</div><div class="stat-lbl">weight banks</div></div>
  <div class="stat-box yellow" id="stream-box" style="display:none">
    <div class="stat-val" id="st-stream">0</div><div class="stat-lbl">stream buf</div>
  </div>
  <div class="stat-box gray"  ><div class="stat-val" id="st-free">0</div><div class="stat-lbl">free banks</div></div>
  <div class="stat-box red"   id="offchip-box" style="display:none">
    <div class="stat-val">OFF</div><div class="stat-lbl">xout → external</div>
  </div>
</div>

<script>
const STATES = {states_json};
const NUM_BANKS = {num_banks};

let current = 0;
const slider   = document.getElementById('layer-slider');
const counter  = document.getElementById('layer-counter');
const bankCont = document.getElementById('bank-container');

// Build bank DOM once
for (let i = 0; i < NUM_BANKS; i++) {{
  const b = document.createElement('div');
  b.className = 'bank free';
  b.id = 'bank-' + i;
  b.innerHTML = '<span class="bank-lbl" id="bl-' + i + '"></span><span class="bank-num">' + i + '</span>';
  bankCont.appendChild(b);
}}

slider.max = STATES.length - 1;

function step(d) {{ setLayer(Math.max(0, Math.min(STATES.length - 1, current + d))); }}

function setLayer(idx) {{
  current = idx;
  slider.value = idx;
  const s = STATES[idx];
  counter.textContent = (idx + 1) + ' / ' + STATES.length + ' — ' + s.name;

  document.getElementById('i-name').textContent    = s.name;
  document.getElementById('i-type').textContent    = s.type;
  document.getElementById('i-input').textContent   = s.input;
  document.getElementById('i-output').textContent  = s.output;
  document.getElementById('i-weights').textContent = s.weights;
  document.getElementById('i-rule').textContent    = s.rule || '—';
  document.getElementById('i-reason').textContent  = s.reason || '—';

  const stayEl = document.getElementById('i-stay');
  if (s.stay) {{
    stayEl.textContent = '✓ On-Chip';
    stayEl.className   = 'rule-pill oncehip-pill';
  }} else {{
    stayEl.textContent = '✗ Off-Chip';
    stayEl.className   = 'rule-pill offchip-pill';
  }}

  // Update banks
  for (let i = 0; i < NUM_BANKS; i++) {{
    const bank = s.banks[i] || {{type:'free', label:''}};
    const el   = document.getElementById('bank-' + i);
    const lbl  = document.getElementById('bl-' + i);
    el.className = 'bank ' + bank.type;
    lbl.textContent = bank.label;
  }}

  // Size bar
  function szSeg(id, banks, label) {{
    const pct = (banks / NUM_BANKS * 100).toFixed(1);
    const el = document.getElementById(id);
    el.style.width   = pct + '%';
    el.textContent   = pct > 5 ? (banks + 'B' + (label ? ' ' + label : '')) : '';
  }}
  szSeg('sz-xin',     s.xin_b,    'xin');
  szSeg('sz-xout',    s.xout_b,   'xout');
  szSeg('sz-xr',      s.xr_b,     'xr');
  szSeg('sz-weights', s.wt_b,     'W');
  szSeg('sz-stream',  s.stream_b, '~');
  szSeg('sz-free',    s.free_b,   'free');

  // Stats
  document.getElementById('st-xin').textContent  = s.xin_b;
  document.getElementById('st-xout').textContent = s.xout_b;
  document.getElementById('st-wt').textContent   = s.wt_b;
  document.getElementById('st-free').textContent = s.free_b;
  const stXr = document.getElementById('st-xr');
  if (stXr) stXr.textContent = s.xr_b;
  document.getElementById('xr-box').style.display     = (s.xr_b > 0) ? '' : 'none';
  const stEl = document.getElementById('st-stream');
  if (stEl) stEl.textContent = s.stream_b;
  document.getElementById('stream-box').style.display  = s.stream_b > 0 ? '' : 'none';
  document.getElementById('offchip-box').style.display = s.stay ? 'none' : '';
}}

setLayer(0);
</script>
</body>
</html>"""


