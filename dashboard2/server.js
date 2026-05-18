import express from 'express'
import cors from 'cors'
import Database from 'better-sqlite3'
import path from 'path'
import fs from 'fs'
import { fileURLToPath } from 'url'
import { spawn } from 'child_process'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const DB_DIR = process.env.DB_DIR || path.join(__dirname, '..', 'runspace', 'database')
const DB_PATH = process.env.DB_PATH || path.join(DB_DIR, 'runs.db')
const OUTPUTS_DIR = path.join(__dirname, '..', 'runspace', 'outputs', 'dashboard_runs')
const LOG_DIR = path.join(DB_DIR, 'logs')
const INPUTS_DIR = path.join(__dirname, '..', 'runspace', 'inputs')
const CONFIGS_DIR = path.join(INPUTS_DIR, 'base_configs')
const REGISTRY_FILE = path.join(OUTPUTS_DIR, 'registry.json')
const PRESETS_FILE = path.join(__dirname, 'presets.json')

function openDB(dbPath) { return new Database(dbPath) }
function getRunsDB() { return openDB(DB_PATH) }

const app = express()
app.use(cors())
app.use(express.json({ limit: '10mb' }))

function parseDT(dt) {
  if (!dt) return { bits: null, exp: null, mant: null }
  if (dt === 'fp32') return { bits: 32, exp: null, mant: null }
  if (dt === 'fp16') return { bits: 16, exp: null, mant: null }
  if (dt === 'bf16') return { bits: 16, exp: null, mant: null }
  if (dt.startsWith('dyn')) return { bits: 0, exp: null, mant: null }
  if (dt.startsWith('int') && !dt.includes('_')) {
    const bits = parseInt(dt.replace('int', ''))
    return { bits: isNaN(bits) ? null : bits, exp: null, mant: null }
  }
  const m = dt.match(/^(?:u?e?fp|int)(\d+)(?:_e(\d+)m(\d+))?/)
  if (m) return { bits: parseInt(m[1]) || null, exp: parseInt(m[2]) || null, mant: parseInt(m[3]) || null }
  return { bits: null, exp: null, mant: null }
}

// ================ HEALTH ================
app.get('/api/health', (_req, res) => {
  res.json({ status: 'ok', db: DB_PATH, db_dir: DB_DIR })
})

// ================ STATS ================
app.get('/api/stats', (req, res) => {
  try {
    const dbPath = req.query.db || DB_PATH
    const db = openDB(dbPath)
    const total = db.prepare('SELECT COUNT(*) as v FROM runs').get().v
    const success = db.prepare("SELECT COUNT(*) as v FROM runs WHERE status='SUCCESS'").get().v
    const models = db.prepare('SELECT COUNT(DISTINCT model_name) as v FROM runs').get().v
    const experiments = db.prepare('SELECT COUNT(DISTINCT experiment_type) as v FROM runs').get().v
    const modelsList = db.prepare('SELECT DISTINCT model_name FROM runs ORDER BY model_name').all().map(r => r.model_name)
    const experimentsList = db.prepare('SELECT DISTINCT experiment_type FROM runs').all().map(r => r.experiment_type)
    const statusesList = db.prepare('SELECT DISTINCT status FROM runs').all().map(r => r.status)
    const weightDTs = db.prepare('SELECT DISTINCT weight_dt FROM runs WHERE weight_dt IS NOT NULL').all().map(r => r.weight_dt)
    const activationDTs = db.prepare('SELECT DISTINCT activation_dt FROM runs WHERE activation_dt IS NOT NULL').all().map(r => r.activation_dt)
    const dateRow = db.prepare("SELECT MIN(run_date) as mn, MAX(run_date) as mx FROM runs WHERE run_date IS NOT NULL").get()
    const minDate = dateRow?.mn || null
    const maxDate = dateRow?.mx || null
    // Datatype bits for filters
    const allDTs = [...new Set([...weightDTs, ...activationDTs])].map(dt => ({ dt, ...parseDT(dt) })).filter(d => d.bits !== null)
    const uniqueBits = [...new Set(allDTs.map(d => d.bits))].sort((a, b) => b - a)

    res.json({
      total, success, models, experiments,
      models_list: modelsList, experiments_list: experimentsList, statuses_list: statusesList,
      weight_dt_list: weightDTs, activation_dt_list: activationDTs,
      min_date: minDate, max_date: maxDate,
      available_bits: uniqueBits,
      dt_info: allDTs,
    })
  } catch (err) {
    res.status(500).json({ error: err.message })
  }
})

// ================ RUNS with full filtering ================
app.get('/api/runs', (req, res) => {
  try {
    const dbPath = req.query.db || DB_PATH
    const db = openDB(dbPath)
    const {
      model, weight_dt, activation_dt, status, experiment,
      w_bits, a_bits,
      sort, order, page, limit, min_date, max_date, newest_only
    } = req.query

    // Lightweight columns only (no giant JSON blobs)
    const cols = 'id, model_name, weight_dt, activation_dt, output_dt, acc1, acc5, ref_acc1, ref_acc5, ref_certainty, experiment_type, run_date, status, mse, l1, certainty'
    let sql = `SELECT ${cols} FROM runs WHERE 1=1`
    const params = []

    function addInFilter(field, value) {
      if (!value) return
      const vals = value.split(',').map(v => v.trim()).filter(Boolean)
      if (vals.length === 0) return
      sql += ` AND ${field} IN (${vals.map(() => '?').join(',')})`
      params.push(...vals)
    }

    addInFilter('model_name', model)
    addInFilter('weight_dt', weight_dt)
    addInFilter('activation_dt', activation_dt)
    addInFilter('status', status)
    addInFilter('experiment_type', experiment)

    if (min_date) { sql += ' AND run_date >= ?'; params.push(min_date) }
    if (max_date) { sql += ' AND run_date <= ?'; params.push(max_date + ' 23:59:59') }

    const allowedSort = ['id', 'model_name', 'weight_dt', 'activation_dt', 'output_dt',
                         'acc1', 'ref_acc1', 'acc5', 'ref_acc5', 'mse', 'l1',
                         'certainty', 'experiment_type', 'status', 'run_date']
    const sortCol = allowedSort.includes(sort) ? sort : 'id'
    const sortDir = order === 'asc' ? 'ASC' : 'DESC'
    const dataSql = sql + ` ORDER BY ${sortCol} ${sortDir}`

    let allRows = db.prepare(dataSql).all(...params)

    // Datatype bits filtering (client-side since we don't store parsed bits)
    if (w_bits) {
      const bits = w_bits.split(',').map(Number).filter(n => !isNaN(n))
      if (bits.length > 0) {
        allRows = allRows.filter(r => bits.includes(parseDT(r.weight_dt).bits))
      }
    }
    if (a_bits) {
      const bits = a_bits.split(',').map(Number).filter(n => !isNaN(n))
      if (bits.length > 0) {
        allRows = allRows.filter(r => bits.includes(parseDT(r.activation_dt).bits))
      }
    }

    if (newest_only === 'true') {
      const seen = new Set()
      allRows = allRows.filter(r => {
        const key = `${r.model_name}|${r.experiment_type}|${r.weight_dt}|${r.activation_dt}`
        if (seen.has(key)) return false
        seen.add(key)
        return true
      })
    }

    const total = allRows.length

    let rows = allRows
    if (limit) {
      const p = Math.max(1, parseInt(page) || 1)
      const l = parseInt(limit)
      rows = rows.slice((p - 1) * l, p * l)
    }

    // Effective references
    if (rows.length > 0) {
      const modelNames = [...new Set(rows.map(r => r.model_name))]
      const refs = db.prepare(`
        SELECT model_name, acc1, acc5, certainty FROM runs
        WHERE model_name IN (${modelNames.map(() => '?').join(',')})
        AND status='SUCCESS' AND experiment_type='fp32_ref'
        ORDER BY id DESC
      `).all(...modelNames)
      const refMap = {}
      for (const ref of refs) {
        if (!refMap[ref.model_name]) {
          refMap[ref.model_name] = { acc1: ref.acc1, acc5: ref.acc5, certainty: ref.certainty }
        }
      }
      for (const r of rows) {
        const ref = refMap[r.model_name]
        r._ref_acc1 = ref?.acc1 ?? r.ref_acc1
        r._ref_acc5 = ref?.acc5 ?? r.ref_acc5
        r._ref_certainty = ref?.certainty ?? r.ref_certainty
      }
    }

    res.json({ total, rows })
  } catch (err) {
    res.status(500).json({ error: err.message })
  }
})

// ================ SINGLE RUN (with full columns) ================
app.get('/api/runs/:id', (req, res) => {
  try {
    const db = getRunsDB()
    const row = db.prepare('SELECT * FROM runs WHERE id = ?').get(Number(req.params.id))
    if (!row) return res.status(404).json({ error: 'Not found' })
    res.json(row)
  } catch (err) {
    res.status(500).json({ error: err.message })
  }
})

// ================ DELETE RUNS ================
app.delete('/api/runs', (req, res) => {
  try {
    const { ids } = req.body
    if (!ids || !Array.isArray(ids) || ids.length === 0) return res.status(400).json({ error: 'ids array required' })
    const db = getRunsDB()
    const placeholders = ids.map(() => '?').join(',')
    db.prepare(`DELETE FROM runs WHERE id IN (${placeholders})`).run(...ids)
    db.close()
    res.json({ deleted: ids.length })
  } catch (err) { res.status(500).json({ error: err.message }) }
})

// ================ UPDATE EXPERIMENT TYPE ================
app.patch('/api/runs/experiment-type', (req, res) => {
  try {
    const { ids, experiment_type } = req.body
    if (!ids || !Array.isArray(ids) || !experiment_type) return res.status(400).json({ error: 'ids and experiment_type required' })
    const db = getRunsDB()
    db.prepare(`UPDATE runs SET experiment_type = ? WHERE id IN (${ids.map(() => '?').join(',')})`).run(experiment_type, ...ids)
    db.close()
    res.json({ updated: ids.length })
  } catch (err) { res.status(500).json({ error: err.message }) }
})

// ================ CREATE DB FROM SELECTION ================
app.post('/api/db/create', (req, res) => {
  try {
    const { ids, name } = req.body
    if (!ids || !Array.isArray(ids) || !name) return res.status(400).json({ error: 'ids and name required' })
    const safeName = name.replace(/[^a-zA-Z0-9\-_\.]/g, '_')
    if (!safeName.endsWith('.db')) safeName += '.db'
    const destPath = path.join(DB_DIR, path.basename(safeName))
    if (fs.existsSync(destPath)) return res.status(409).json({ error: 'File exists' })
    const srcDb = getRunsDB()
    const rows = srcDb.prepare(`SELECT * FROM runs WHERE id IN (${ids.map(() => '?').join(',')})`).all(...ids)
    srcDb.close()
    const destDb = new Database(destPath)
    destDb.exec(`CREATE TABLE IF NOT EXISTS runs (
      id INTEGER PRIMARY KEY AUTOINCREMENT, model_name TEXT NOT NULL,
      weight_dt TEXT, activation_dt TEXT, output_dt TEXT,
      acc1 REAL, acc5 REAL, ref_acc1 REAL, ref_acc5 REAL, ref_certainty REAL,
      experiment_type TEXT, run_date TEXT, status TEXT,
      mse REAL, l1 REAL, certainty REAL, cli_command TEXT,
      quant_map_json TEXT, input_map_json TEXT, output_map_json TEXT, config_json TEXT
    )`)
    const insert = destDb.prepare(`INSERT INTO runs (model_name, weight_dt, activation_dt, output_dt,
      acc1, acc5, ref_acc1, ref_acc5, ref_certainty, experiment_type, run_date, status,
      mse, l1, certainty, cli_command, quant_map_json, input_map_json, output_map_json, config_json)
      VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`)
    const insertMany = destDb.transaction((rws) => {
      for (const r of rws) insert.run(r.model_name, r.weight_dt, r.activation_dt, r.output_dt,
        r.acc1, r.acc5, r.ref_acc1, r.ref_acc5, r.ref_certainty, r.experiment_type, r.run_date, r.status,
        r.mse, r.l1, r.certainty, r.cli_command, r.quant_map_json, r.input_map_json, r.output_map_json, r.config_json)
    })
    insertMany(rows)
    destDb.close()
    res.json({ created: destPath, rows: rows.length })
  } catch (err) { res.status(500).json({ error: err.message }) }
})

// ================ DB MANAGEMENT ================
app.get('/api/db/list', (_req, res) => {
  try {
    if (!fs.existsSync(DB_DIR)) return res.json([])
    const files = fs.readdirSync(DB_DIR).filter(f => f.endsWith('.db')).map(f => {
      const fp = path.join(DB_DIR, f)
      const stat = fs.statSync(fp)
      return { name: f, path: fp, size: stat.size, mtime: stat.mtime.toISOString() }
    }).sort((a, b) => b.mtime.localeCompare(a.mtime))
    res.json(files)
  } catch (err) { res.status(500).json({ error: err.message }) }
})

app.post('/api/db/rename', (req, res) => {
  try {
    const { oldName, newName } = req.body
    if (!oldName || !newName) return res.status(400).json({ error: 'oldName and newName required' })
    const safe = path.basename(newName.replace(/[^a-zA-Z0-9\-_\.]/g, '_'))
    const on = path.join(DB_DIR, path.basename(oldName))
    const nn = path.join(DB_DIR, safe.endsWith('.db') ? safe : safe + '.db')
    if (!fs.existsSync(on)) return res.status(404).json({ error: 'Not found' })
    if (fs.existsSync(nn)) return res.status(409).json({ error: 'Target exists' })
    fs.renameSync(on, nn)
    res.json({ from: on, to: nn })
  } catch (err) { res.status(500).json({ error: err.message }) }
})

app.delete('/api/db/delete', (req, res) => {
  try {
    const { name } = req.body
    if (!name) return res.status(400).json({ error: 'name required' })
    const fp = path.join(DB_DIR, path.basename(name))
    if (!fs.existsSync(fp)) return res.status(404).json({ error: 'Not found' })
    fs.unlinkSync(fp)
    res.json({ deleted: fp })
  } catch (err) { res.status(500).json({ error: err.message }) }
})

// ================ CACHE SIMULATIONS ================
app.get('/api/cache-simulations', (req, res) => {
  try {
    const dbPath = req.query.db || DB_PATH
    const db = openDB(dbPath)
    const has = db.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='cache_simulations'").get()
    if (!has) return res.json([])
    const rows = db.prepare('SELECT * FROM cache_simulations ORDER BY id DESC').all()
    db.close()
    res.json(rows)
  } catch (err) { res.status(500).json({ error: err.message }) }
})

// ================ MODEL GRAPHS ================
app.get('/api/model-graphs', (req, res) => {
  try {
    const dbPath = req.query.db || DB_PATH
    const db = openDB(dbPath)
    const has = db.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='model_graphs'").get()
    if (!has) return res.json([])
    const model = req.query.model
    const rows = model
      ? db.prepare('SELECT * FROM model_graphs WHERE model_name = ?').all(model)
      : db.prepare('SELECT model_name, generated_at FROM model_graphs ORDER BY generated_at DESC').all()
    db.close()
    res.json(rows)
  } catch (err) { res.status(500).json({ error: err.message }) }
})

// ================ FM RUNS ================
app.get('/api/fm-runs', (req, res) => {
  try {
    const dbPath = req.query.db || path.join(DB_DIR, 'fm_runs.db')
    if (!fs.existsSync(dbPath)) return res.json({ total: 0, rows: [] })
    const db = openDB(dbPath)
    const has = db.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='fm_runs'").get()
    if (!has) return res.json({ total: 0, rows: [] })
    const { model, sort, order, page, limit, status } = req.query
    let sql = 'SELECT * FROM fm_runs WHERE 1=1'
    const params = []
    if (model) {
      const models = model.split(',').map(m => m.trim()).filter(Boolean)
      if (models.length > 0) {
        sql += ` AND model_name IN (${models.map(() => '?').join(',')})`
        params.push(...models)
      }
    }
    if (status) {
      const statuses = status.split(',').map(s => s.trim()).filter(Boolean)
      if (statuses.length > 0) {
        sql += ` AND status IN (${statuses.map(() => '?').join(',')})`
        params.push(...statuses)
      }
    }
    const allowedSort = ['id', 'model_name', 'matching_score', 'pose_auc_5', 'pose_auc_10', 'pose_auc_20']
    const sortCol = allowedSort.includes(sort) ? sort : 'id'
    sql += ` ORDER BY ${sortCol} ${order === 'asc' ? 'ASC' : 'DESC'}`
    const total = db.prepare(`SELECT COUNT(*) as total FROM (${sql})`).get(...params).total
    if (limit) {
      const p = Math.max(1, parseInt(page) || 1)
      sql += ` LIMIT ? OFFSET ?`
      params.push(parseInt(limit), (p - 1) * parseInt(limit))
    }
    const rows = db.prepare(sql).all(...params)
    db.close()
    res.json({ total, rows })
  } catch (err) { res.status(500).json({ error: err.message }) }
})

// ================ COMPARE ================
app.get('/api/compare', (req, res) => {
  try {
    const dbPath = req.query.db || DB_PATH
    const db = openDB(dbPath)
    const refs = db.prepare("SELECT model_name, acc1 as ref_acc, acc5 as ref_acc5 FROM runs WHERE experiment_type='fp32_ref' AND status='SUCCESS'").all()
    const quants = db.prepare("SELECT id, model_name, weight_dt, activation_dt, output_dt, acc1, acc5, experiment_type FROM runs WHERE experiment_type!='fp32_ref' AND status='SUCCESS'").all()
    db.close()
    res.json({ refs, quants })
  } catch (err) { res.status(500).json({ error: err.message }) }
})

// ================ CONFIGS / MODELS ================
app.get('/api/configs', (_req, res) => {
  try {
    const configs = []
    for (const dir of [CONFIGS_DIR, INPUTS_DIR]) {
      if (fs.existsSync(dir)) {
        for (const f of fs.readdirSync(dir)) {
          if ((f.endsWith('.yaml') || f.endsWith('.yml')) && !f.includes('models')) {
            configs.push({ name: f, path: path.join(dir, f) })
          }
        }
      }
    }
    res.json(configs)
  } catch (err) { res.status(500).json({ error: err.message }) }
})

app.get('/api/models', (_req, res) => {
  try {
    const mp = path.join(INPUTS_DIR, 'models.yaml')
    const models = []
    if (fs.existsSync(mp)) {
      const content = fs.readFileSync(mp, 'utf-8')
      for (const m of content.matchAll(/^\s*-\s+?(?:name:\s*)?['"`]?(\S+?)['"`]?\s*$/gm)) {
        if (m[1] && !m[1].startsWith('#')) models.push(m[1])
      }
      // Also try without name: prefix
      if (models.length === 0) {
        for (const m of content.matchAll(/^\s*-\s+['"`]?(\S+?)['"`]?\s*$/gm)) {
          if (m[1] && !m[1].startsWith('#') && m[1] !== '---') models.push(m[1])
        }
      }
    }
    if (models.length === 0) models.push('resnet18', 'resnet50', 'mobilenet_v3_large', 'efficientnet_b0', 'vit_b_16', 'mobilevit_s')
    res.json(models)
  } catch (err) { res.status(500).json({ error: err.message }) }
})

// ================ PRESETS ================
app.get('/api/presets', (_req, res) => {
  try {
    if (fs.existsSync(PRESETS_FILE)) {
      res.json(JSON.parse(fs.readFileSync(PRESETS_FILE, 'utf-8')))
    } else {
      res.json({})
    }
  } catch (err) { res.status(500).json({ error: err.message }) }
})

app.post('/api/presets', (req, res) => {
  try {
    fs.writeFileSync(PRESETS_FILE, JSON.stringify(req.body, null, 2))
    res.json({ saved: true })
  } catch (err) { res.status(500).json({ error: err.message }) }
})

// ================ RUN LAUNCHER ================
function loadRegistry() {
  try { if (fs.existsSync(REGISTRY_FILE)) return JSON.parse(fs.readFileSync(REGISTRY_FILE, 'utf-8')) } catch {}
  return []
}
function saveRegistry(r) {
  if (!fs.existsSync(OUTPUTS_DIR)) fs.mkdirSync(OUTPUTS_DIR, { recursive: true })
  fs.writeFileSync(REGISTRY_FILE, JSON.stringify(r, null, 2))
}
function isPidRunning(pid) {
  try { return fs.readFileSync(`/proc/${pid}/stat`, 'utf-8').split(' ')[2] !== 'Z' } catch { return false }
}

app.get('/api/run-launcher/status', (_req, res) => {
  try {
    const reg = loadRegistry()
    for (const r of reg) {
      if (r.status === 'running' && !isPidRunning(r.pid)) r.status = 'finished'
    }
    saveRegistry(reg)
    res.json(reg)
  } catch (err) { res.status(500).json({ error: err.message }) }
})

app.post('/api/run-launcher/launch', (req, res) => {
  try {
    const { command, name } = req.body
    if (!command) return res.status(400).json({ error: 'command required' })
    if (!fs.existsSync(LOG_DIR)) fs.mkdirSync(LOG_DIR, { recursive: true })
    const ts = new Date().toISOString().slice(0, 19).replace(/[-:T]/g, '').slice(0, 12)
    const safe = (name || 'run').replace(/[^a-zA-Z0-9\-_]/g, '_')
    const logPath = path.join(LOG_DIR, `${ts}_${safe}.log`)
    const logStream = fs.createWriteStream(logPath)
    logStream.write(`=== QBench Run ===\nDate: ${new Date().toISOString()}\nCommand: ${command}\n===\n\n`)
    const child = spawn('bash', ['-c', command], {
      cwd: path.join(__dirname, '..'),
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
      detached: true, stdio: ['ignore', 'pipe', 'pipe'],
    })
    child.stdout.pipe(logStream)
    child.stderr.pipe(logStream)
    const reg = loadRegistry()
    const entry = { pid: child.pid, status: 'running', command, name: safe, log_path: logPath, started: new Date().toISOString(), exit_code: null }
    reg.push(entry)
    saveRegistry(reg)
    child.on('close', (code) => {
      const r2 = loadRegistry()
      const idx = r2.findIndex(e => e.pid === child.pid)
      if (idx >= 0) { r2[idx].status = 'finished'; r2[idx].exit_code = code; saveRegistry(r2) }
    })
    res.json({ pid: child.pid, log_path: logPath, status: 'launched' })
  } catch (err) { res.status(500).json({ error: err.message }) }
})

app.get('/api/run-launcher/logs/:name', (req, res) => {
  try {
    const lp = path.join(LOG_DIR, path.basename(req.params.name))
    if (!fs.existsSync(lp)) return res.status(404).json({ error: 'Not found' })
    const { tail, offset, lines } = req.query
    const stat = fs.statSync(lp)
    let content
    if (tail) {
      const tb = parseInt(tail)
      const start = Math.max(0, stat.size - tb)
      const fd = fs.openSync(lp, 'r')
      const buf = Buffer.alloc(Math.min(stat.size - start, tb))
      fs.readSync(fd, buf, 0, buf.length, start)
      fs.closeSync(fd)
      content = buf.toString('utf-8')
    } else if (offset !== undefined) {
      const o = parseInt(offset) || 0
      const l = parseInt(lines) || 1000
      const fd = fs.openSync(lp, 'r')
      const buf = Buffer.alloc(Math.min(stat.size - o, 1024 * 1024))
      fs.readSync(fd, buf, 0, buf.length, o)
      fs.closeSync(fd)
      content = buf.toString('utf-8').split('\n').slice(0, l).join('\n')
    } else {
      content = fs.readFileSync(lp, 'utf-8')
    }
    res.json({ content, size: stat.size, path: lp })
  } catch (err) { res.status(500).json({ error: err.message }) }
})

// ================ WIN RATE ANALYSIS ================
function parseQuantMap(raw) {
  if (!raw) return null
  let parsed
  try { parsed = JSON.parse(raw) } catch { return null }
  if (!parsed || (!Array.isArray(parsed) && typeof parsed !== 'object')) return null
  if (!Array.isArray(parsed)) {
    const keys = Object.keys(parsed).sort((a, b) => {
      const na = parseInt(a.match(/\d+/)?.[0] || '0'), nb = parseInt(b.match(/\d+/)?.[0] || '0')
      return na - nb
    })
    return keys.map((k, i) => {
      const d = parsed[k]
      return typeof d === 'object' && d !== null ? { layer: k, index: i, ...d } : { layer: k, index: i, format: String(d), type: '', chunks: 1 }
    })
  }
  return parsed.map((item, i) => typeof item === 'string' ? { layer: `layer_${i}`, index: i, format: String(item), type: '', chunks: 1 } : { layer: item.layer || `layer_${i}`, index: i, ...item })
}

app.post('/api/analyze/win-rates', (req, res) => {
  try {
    const { quant_map_json } = req.body
    const layers = parseQuantMap(quant_map_json)
    if (!layers || layers.length === 0) return res.json(null)
    const formatCounts = {}
    let totalChunks = 0
    const layerDetails = []
    for (const l of layers) {
      const fmt = l.dominant_format || l.format || 'unknown'
      const chunks = l.total_chunks || l.chunks || 1
      totalChunks += chunks
      formatCounts[fmt] = (formatCounts[fmt] || 0) + chunks
      layerDetails.push({ layer: l.layer, index: l.index, type: l.type || '', dominant_format: fmt, chunks })
    }
    const totalLayers = layers.length
    const summary = Object.entries(formatCounts).map(([format, chunks]) => ({
      format, layer_wins: layerDetails.filter(l => l.dominant_format === format).length,
      layer_win_rate: ((layerDetails.filter(l => l.dominant_format === format).length / totalLayers) * 100).toFixed(1),
      chunk_wins: chunks, chunk_win_rate: ((chunks / totalChunks) * 100).toFixed(1),
    })).sort((a, b) => b.chunk_wins - a.chunk_wins)
    res.json({ summary, layers: layerDetails, meta: { total_layers: totalLayers, total_chunks: totalChunks, top_format: summary[0]?.format, unique_formats: Object.keys(formatCounts).length } })
  } catch (err) { res.status(500).json({ error: err.message }) }
})

// ================ STATIC ================
const distPath = path.join(__dirname, 'dist')
app.use(express.static(distPath))
app.get('*', (_req, res) => res.sendFile(path.join(distPath, 'index.html')))

const PORT = process.env.PORT || 3000
app.listen(PORT, '0.0.0.0', () => {
  console.log(`QBench Dashboard v2 running on http://0.0.0.0:${PORT}`)
  console.log(`Database: ${DB_PATH}`)
})
