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

function openDB(dbPath) {
  return new Database(dbPath)
}

function getRunsDB() {
  return openDB(DB_PATH)
}

const app = express()
app.use(cors())
app.use(express.json({ limit: '10mb' }))

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

    const weightBits = db.prepare('SELECT DISTINCT weight_dt FROM runs WHERE weight_dt IS NOT NULL').all().map(r => r.weight_dt)
    const activationBits = db.prepare('SELECT DISTINCT activation_dt FROM runs WHERE activation_dt IS NOT NULL').all().map(r => r.activation_dt)

    const dateRow = db.prepare("SELECT MIN(run_date) as mn, MAX(run_date) as mx FROM runs WHERE run_date IS NOT NULL").get()
    const minDate = dateRow?.mn || null
    const maxDate = dateRow?.mx || null

    res.json({
      total, success, models, experiments,
      models_list: modelsList,
      experiments_list: experimentsList,
      statuses_list: statusesList,
      weight_formats: weightBits,
      activation_formats: activationBits,
      min_date: minDate,
      max_date: maxDate,
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
    const { model, weight_bits, activation_bits, status, experiment,
            sort, order, page, limit, min_date, max_date, newest_only } = req.query

    let sql = 'SELECT id, model_name, weight_dt, activation_dt, output_dt, acc1, acc5, ref_acc1, ref_acc5, ref_certainty, experiment_type, run_date, status, mse, l1, certainty FROM runs WHERE 1=1'
    const params = []

    if (model) {
      const models = model.split(',').map(m => m.trim()).filter(Boolean)
      if (models.length > 0) {
        sql += ` AND model_name IN (${models.map(() => '?').join(',')})`
        params.push(...models)
      }
    }
    if (weight_bits) {
      const bits = weight_bits.split(',').map(b => b.trim()).filter(Boolean)
      if (bits.length > 0) {
        sql += ` AND weight_dt IN (${bits.map(() => '?').join(',')})`
        params.push(...bits)
      }
    }
    if (activation_bits) {
      const bits = activation_bits.split(',').map(b => b.trim()).filter(Boolean)
      if (bits.length > 0) {
        sql += ` AND activation_dt IN (${bits.map(() => '?').join(',')})`
        params.push(...bits)
      }
    }
    if (status && status !== 'all') {
      const statuses = status.split(',').map(s => s.trim()).filter(Boolean)
      if (statuses.length > 0) {
        sql += ` AND status IN (${statuses.map(() => '?').join(',')})`
        params.push(...statuses)
      }
    }
    if (experiment && experiment !== 'all') {
      const experiments = experiment.split(',').map(e => e.trim()).filter(Boolean)
      if (experiments.length > 0) {
        sql += ` AND experiment_type IN (${experiments.map(() => '?').join(',')})`
        params.push(...experiments)
      }
    }
    if (min_date) {
      sql += ' AND run_date >= ?'
      params.push(min_date)
    }
    if (max_date) {
      sql += ' AND run_date <= ?'
      params.push(max_date + ' 23:59:59')
    }

    const allowedSort = ['id', 'model_name', 'weight_dt', 'activation_dt', 'output_dt',
                         'acc1', 'ref_acc1', 'acc5', 'ref_acc5', 'mse', 'l1',
                         'certainty', 'experiment_type', 'status', 'run_date']
    const sortCol = allowedSort.includes(sort) ? sort : 'id'
    const sortDir = order === 'asc' ? 'ASC' : 'DESC'
    const dataSql = sql + ` ORDER BY ${sortCol} ${sortDir}`

    const allRows = db.prepare(dataSql).all(...params)

    let filteredRows = allRows
    if (newest_only === 'true') {
      const seen = new Set()
      filteredRows = allRows.filter(r => {
        const key = `${r.model_name}|${r.experiment_type}|${r.weight_dt}|${r.activation_dt}`
        if (seen.has(key)) return false
        seen.add(key)
        return true
      })
    }

    const total = filteredRows.length

    let rows = filteredRows
    if (limit) {
      const p = Math.max(1, parseInt(page) || 1)
      const l = parseInt(limit)
      rows = rows.slice((p - 1) * l, p * l)
    }

    // Get effective references for non-fp32-ref runs
    const refMap = {}
    if (rows.length > 0) {
      const modelNames = [...new Set(rows.map(r => r.model_name))]
      const refs = db.prepare(`
        SELECT model_name, acc1, acc5, certainty FROM runs
        WHERE model_name IN (${modelNames.map(() => '?').join(',')})
        AND status='SUCCESS'
        AND experiment_type='fp32_ref'
        ORDER BY id DESC
      `).all(...modelNames)
      for (const ref of refs) {
        if (!refMap[ref.model_name]) {
          refMap[ref.model_name] = { ref_acc1: ref.acc1, ref_acc5: ref.acc5, ref_certainty: ref.certainty }
        }
      }
    }

    res.json({
      total,
      rows: rows.map(r => ({
        ...r,
        _ref_acc1: refMap[r.model_name]?.ref_acc1 ?? r.ref_acc1,
        _ref_acc5: refMap[r.model_name]?.ref_acc5 ?? r.ref_acc5,
        _ref_certainty: refMap[r.model_name]?.ref_certainty ?? r.ref_certainty,
      })),
    })
  } catch (err) {
    res.status(500).json({ error: err.message })
  }
})

// ================ SINGLE RUN ================
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
    if (!ids || !Array.isArray(ids) || ids.length === 0) {
      return res.status(400).json({ error: 'ids array required' })
    }
    const db = getRunsDB()
    const placeholders = ids.map(() => '?').join(',')
    const result = db.prepare(`DELETE FROM runs WHERE id IN (${placeholders})`).run(...ids)
    db.close()
    res.json({ deleted: result.changes })
  } catch (err) {
    res.status(500).json({ error: err.message })
  }
})

// ================ UPDATE EXPERIMENT TYPE ================
app.patch('/api/runs/experiment-type', (req, res) => {
  try {
    const { ids, experiment_type } = req.body
    if (!ids || !Array.isArray(ids) || ids.length === 0 || !experiment_type) {
      return res.status(400).json({ error: 'ids array and experiment_type required' })
    }
    const db = getRunsDB()
    const placeholders = ids.map(() => '?').join(',')
    const result = db.prepare(`UPDATE runs SET experiment_type = ? WHERE id IN (${placeholders})`).run(experiment_type, ...ids)
    db.close()
    res.json({ updated: result.changes })
  } catch (err) {
    res.status(500).json({ error: err.message })
  }
})

// ================ CREATE DB FROM SELECTION ================
app.post('/api/db/create', (req, res) => {
  try {
    const { ids, name } = req.body
    if (!ids || !Array.isArray(ids) || ids.length === 0 || !name) {
      return res.status(400).json({ error: 'ids array and name required' })
    }
    const safeName = name.replace(/[^a-zA-Z0-9\-_\.]/g, '_')
    if (!safeName.endsWith('.db')) safeName += '.db'
    const destPath = path.join(DB_DIR, path.basename(safeName))
    if (fs.existsSync(destPath)) {
      return res.status(409).json({ error: 'Database file already exists' })
    }

    const srcDb = getRunsDB()
    const placeholders = ids.map(() => '?').join(',')
    const rows = srcDb.prepare(`SELECT * FROM runs WHERE id IN (${placeholders})`).all(...ids)
    srcDb.close()

    const destDb = new Database(destPath)
    destDb.exec(`CREATE TABLE IF NOT EXISTS runs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      model_name TEXT NOT NULL,
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
      for (const r of rws) {
        insert.run(r.model_name, r.weight_dt, r.activation_dt, r.output_dt,
          r.acc1, r.acc5, r.ref_acc1, r.ref_acc5, r.ref_certainty,
          r.experiment_type, r.run_date, r.status,
          r.mse, r.l1, r.certainty, r.cli_command,
          r.quant_map_json, r.input_map_json, r.output_map_json, r.config_json)
      }
    })
    insertMany(rows)
    destDb.close()
    res.json({ created: destPath, rows: rows.length })
  } catch (err) {
    res.status(500).json({ error: err.message })
  }
})

// ================ DB MANAGEMENT ================
app.get('/api/db/list', (_req, res) => {
  try {
    const files = fs.readdirSync(DB_DIR)
      .filter(f => f.endsWith('.db'))
      .map(f => {
        const fp = path.join(DB_DIR, f)
        const stat = fs.statSync(fp)
        return { name: f, path: fp, size: stat.size, mtime: stat.mtime.toISOString() }
      })
      .sort((a, b) => b.mtime.localeCompare(a.mtime))
    res.json(files)
  } catch (err) {
    res.status(500).json({ error: err.message })
  }
})

app.post('/api/db/rename', (req, res) => {
  try {
    const { oldName, newName } = req.body
    if (!oldName || !newName) return res.status(400).json({ error: 'oldName and newName required' })
    const safeNewName = path.basename(newName.replace(/[^a-zA-Z0-9\-_\.]/g, '_'))
    if (!safeNewName.endsWith('.db')) safeNewName += '.db'
    const oldPath = path.join(DB_DIR, path.basename(oldName))
    const newPath = path.join(DB_DIR, safeNewName)
    if (!fs.existsSync(oldPath)) return res.status(404).json({ error: 'Source DB not found' })
    if (fs.existsSync(newPath)) return res.status(409).json({ error: 'Target DB already exists' })
    fs.renameSync(oldPath, newPath)
    res.json({ renamed: { from: oldPath, to: newPath } })
  } catch (err) {
    res.status(500).json({ error: err.message })
  }
})

app.delete('/api/db/delete', (req, res) => {
  try {
    const { name } = req.body
    if (!name) return res.status(400).json({ error: 'name required' })
    const dbPath = path.join(DB_DIR, path.basename(name))
    if (!fs.existsSync(dbPath)) return res.status(404).json({ error: 'DB not found' })
    fs.unlinkSync(dbPath)
    res.json({ deleted: dbPath })
  } catch (err) {
    res.status(500).json({ error: err.message })
  }
})

// ================ CACHE SIMULATIONS ================
app.get('/api/cache-simulations', (req, res) => {
  try {
    const dbPath = req.query.db || DB_PATH
    const db = openDB(dbPath)
    const hasTable = db.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='cache_simulations'").get()
    if (!hasTable) return res.json([])
    const rows = db.prepare('SELECT * FROM cache_simulations ORDER BY id DESC').all()
    db.close()
    res.json(rows)
  } catch (err) {
    res.status(500).json({ error: err.message })
  }
})

// ================ MODEL GRAPHS ================
app.get('/api/model-graphs', (req, res) => {
  try {
    const dbPath = req.query.db || DB_PATH
    const db = openDB(dbPath)
    const hasTable = db.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='model_graphs'").get()
    if (!hasTable) return res.json([])
    const model = req.query.model
    let rows
    if (model) {
      rows = db.prepare('SELECT * FROM model_graphs WHERE model_name = ?').all(model)
    } else {
      rows = db.prepare('SELECT model_name, generated_at FROM model_graphs ORDER BY generated_at DESC').all()
    }
    db.close()
    res.json(rows)
  } catch (err) {
    res.status(500).json({ error: err.message })
  }
})

// ================ FEATURE MATCHING RUNS ================
app.get('/api/fm-runs', (req, res) => {
  try {
    const dbPath = req.query.db || path.join(DB_DIR, 'fm_runs.db')
    if (!fs.existsSync(dbPath)) return res.json({ total: 0, rows: [] })
    const db = openDB(dbPath)
    const hasTable = db.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='fm_runs'").get()
    if (!hasTable) return res.json({ total: 0, rows: [] })
    const { sort, order, page, limit } = req.query
    let sql = 'SELECT * FROM fm_runs'
    const allowedSort = ['id', 'model_name', 'matching_score', 'pose_auc_5', 'pose_auc_10', 'pose_auc_20']
    const sortCol = allowedSort.includes(sort) ? sort : 'id'
    const sortDir = order === 'asc' ? 'ASC' : 'DESC'
    sql += ` ORDER BY ${sortCol} ${sortDir}`
    const total = db.prepare('SELECT COUNT(*) as total FROM fm_runs').get().total
    const params = []
    if (limit) {
      const p = Math.max(1, parseInt(page) || 1)
      const l = parseInt(limit)
      sql += ` LIMIT ? OFFSET ?`
      params.push(l, (p - 1) * l)
    }
    const rows = db.prepare(sql).all(...params)
    db.close()
    res.json({ total, rows })
  } catch (err) {
    res.status(500).json({ error: err.message })
  }
})

// ================ COMPARE ================
app.get('/api/compare', (req, res) => {
  try {
    const dbPath = req.query.db || DB_PATH
    const db = openDB(dbPath)
    const refRows = db.prepare(`
      SELECT model_name, acc1 as ref_acc, acc5 as ref_acc5
      FROM runs WHERE experiment_type='fp32_ref' AND status='SUCCESS'
    `).all()
    const quantRows = db.prepare(`
      SELECT model_name, weight_dt, activation_dt, output_dt, acc1, acc5, experiment_type
      FROM runs WHERE experiment_type!='fp32_ref' AND status='SUCCESS'
    `).all()
    db.close()
    res.json({ refs: refRows, quants: quantRows })
  } catch (err) {
    res.status(500).json({ error: err.message })
  }
})

// ================ CONFIGS ================
app.get('/api/configs', (_req, res) => {
  try {
    const configs = []
    if (fs.existsSync(CONFIGS_DIR)) {
      for (const f of fs.readdirSync(CONFIGS_DIR)) {
        if (f.endsWith('.yaml') || f.endsWith('.yml')) {
          configs.push({ name: f, path: path.join(CONFIGS_DIR, f) })
        }
      }
    }
    // Also check inputs/ directory
    if (fs.existsSync(INPUTS_DIR)) {
      for (const f of fs.readdirSync(INPUTS_DIR)) {
        if ((f.endsWith('.yaml') || f.endsWith('.yml')) && f !== 'models.yaml') {
          configs.push({ name: f, path: path.join(INPUTS_DIR, f) })
        }
      }
    }
    res.json(configs)
  } catch (err) {
    res.status(500).json({ error: err.message })
  }
})

// ================ MODELS ================
app.get('/api/models', (_req, res) => {
  try {
    const modelsPath = path.join(INPUTS_DIR, 'models.yaml')
    const models = []
    if (fs.existsSync(modelsPath)) {
      const content = fs.readFileSync(modelsPath, 'utf-8')
      // Simple YAML model name extraction
      const matches = content.matchAll(/^\s*-\s+?(?:name:\s*)?(\S+)/gm)
      for (const m of matches) {
        if (m[1] && !m[1].startsWith('#')) models.push(m[1].replace(/['"]/g, ''))
      }
    }
    if (models.length === 0) {
      models.push('resnet18', 'resnet50', 'mobilenet_v3_large', 'efficientnet_b0', 'vit_b_16', 'mobilevit_s')
    }
    res.json(models)
  } catch (err) {
    res.status(500).json({ error: err.message })
  }
})

// ================ RUN LAUNCHER ================
function loadRegistry() {
  try {
    if (fs.existsSync(REGISTRY_FILE)) {
      return JSON.parse(fs.readFileSync(REGISTRY_FILE, 'utf-8'))
    }
  } catch {}
  return []
}

function saveRegistry(registry) {
  if (!fs.existsSync(OUTPUTS_DIR)) fs.mkdirSync(OUTPUTS_DIR, { recursive: true })
  fs.writeFileSync(REGISTRY_FILE, JSON.stringify(registry, null, 2))
}

function isPidRunning(pid) {
  try {
    const stat = fs.readFileSync(`/proc/${pid}/stat`, 'utf-8')
    const state = stat.split(' ')[2]
    return state !== 'Z'
  } catch { return false }
}

app.get('/api/run-launcher/status', (_req, res) => {
  try {
    const registry = loadRegistry()
    const running = registry.filter(r => r.status === 'running')
    for (const r of running) {
      if (!isPidRunning(r.pid)) {
        r.status = 'finished'
      }
    }
    if (running.some(r => r.status === 'finished')) saveRegistry(registry)
    res.json(registry)
  } catch (err) {
    res.status(500).json({ error: err.message })
  }
})

app.post('/api/run-launcher/launch', (req, res) => {
  try {
    const { command, name } = req.body
    if (!command) return res.status(400).json({ error: 'command required' })

    if (!fs.existsSync(LOG_DIR)) fs.mkdirSync(LOG_DIR, { recursive: true })
    const timestamp = new Date().toISOString().slice(0, 19).replace(/[-:T]/g, '').slice(0, 12)
    const safeName = (name || 'dashboard_run').replace(/[^a-zA-Z0-9\-_]/g, '_')
    const logName = `${timestamp}_${safeName}.log`
    const logPath = path.join(LOG_DIR, logName)

    const logStream = fs.createWriteStream(logPath)
    logStream.write(`=== QBench Dashboard Run ===\n`)
    logStream.write(`Date: ${new Date().toISOString()}\n`)
    logStream.write(`Command: ${command}\n`)
    logStream.write(`========================================\n\n`)

    const child = spawn('bash', ['-c', command], {
      cwd: path.join(__dirname, '..'),
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
      detached: true,
      stdio: ['ignore', 'pipe', 'pipe'],
    })

    child.stdout.pipe(logStream)
    child.stderr.pipe(logStream)

    const registry = loadRegistry()
    const entry = {
      pid: child.pid,
      status: 'running',
      command,
      name: safeName,
      log_path: logPath,
      started: new Date().toISOString(),
      exit_code: null,
    }
    registry.push(entry)
    saveRegistry(registry)

    child.on('close', (code) => {
      const reg = loadRegistry()
      const idx = reg.findIndex(r => r.pid === child.pid)
      if (idx >= 0) {
        reg[idx].status = 'finished'
        reg[idx].exit_code = code
        saveRegistry(reg)
      }
    })

    res.json({ pid: child.pid, log_path: logPath, status: 'launched' })
  } catch (err) {
    res.status(500).json({ error: err.message })
  }
})

app.get('/api/run-launcher/logs/:name', (req, res) => {
  try {
    const logPath = path.join(LOG_DIR, path.basename(req.params.name))
    if (!fs.existsSync(logPath)) return res.status(404).json({ error: 'Log not found' })
    const { tail } = req.query
    const stat = fs.statSync(logPath)
    let content
    if (tail) {
      const tailBytes = parseInt(tail) || 131072
      const start = Math.max(0, stat.size - tailBytes)
      const fd = fs.openSync(logPath, 'r')
      const buf = Buffer.alloc(Math.min(stat.size - start, tailBytes))
      fs.readSync(fd, buf, 0, buf.length, start)
      fs.closeSync(fd)
      content = buf.toString('utf-8')
    } else {
      content = fs.readFileSync(logPath, 'utf-8')
    }
    res.json({ content, size: stat.size, path: logPath })
  } catch (err) {
    res.status(500).json({ error: err.message })
  }
})

// ================ WIN RATE ANALYSIS ================
function parseQuantMap(raw) {
  if (!raw) return null
  let parsed
  try { parsed = JSON.parse(raw) } catch { return null }
  if (!parsed || (!Array.isArray(parsed) && typeof parsed !== 'object')) return null
  // Normalize: if object keyed by layer name, convert to array
  if (!Array.isArray(parsed)) {
    const layers = []
    // Sort by key to get topological order
    const keys = Object.keys(parsed).sort((a, b) => {
      const na = parseInt(a.match(/\d+/)?.[0] || '0')
      const nb = parseInt(b.match(/\d+/)?.[0] || '0')
      return na - nb
    })
    for (let i = 0; i < keys.length; i++) {
      const layerData = parsed[keys[i]]
      if (typeof layerData === 'string' || typeof layerData === 'number') {
        layers.push({ layer: keys[i], index: i, format: String(layerData), type: '', chunks: 1 })
      } else if (typeof layerData === 'object' && layerData !== null) {
        layers.push({ layer: keys[i], index: i, ...layerData })
      }
    }
    return layers
  }
  return parsed.map((item, i) => {
    if (typeof item === 'string' || typeof item === 'number') {
      return { layer: `layer_${i}`, index: i, format: String(item), type: '', chunks: 1 }
    }
    return { layer: item.layer || `layer_${i}`, index: i, ...item }
  })
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
      const type = l.type || ''
      const chunks = l.total_chunks || l.chunks || 1
      totalChunks += chunks
      formatCounts[fmt] = (formatCounts[fmt] || 0) + chunks

      layerDetails.push({
        layer: l.layer,
        index: l.index,
        type,
        dominant_format: fmt,
        chunks,
      })
    }

    const totalLayers = layers.length
    const summary = Object.entries(formatCounts)
      .map(([format, chunks]) => ({
        format,
        layer_wins: layerDetails.filter(l => l.dominant_format === format).length,
        layer_win_rate: ((layerDetails.filter(l => l.dominant_format === format).length / totalLayers) * 100).toFixed(1),
        chunk_wins: chunks,
        chunk_win_rate: ((chunks / totalChunks) * 100).toFixed(1),
      }))
      .sort((a, b) => b.chunk_wins - a.chunk_wins)

    const topFormat = summary.length > 0 ? summary[0].format : null

    res.json({
      summary,
      layers: layerDetails,
      meta: {
        total_layers: totalLayers,
        total_chunks: totalChunks,
        top_format: topFormat,
        unique_formats: Object.keys(formatCounts).length,
      },
    })
  } catch (err) {
    res.status(500).json({ error: err.message })
  }
})

// ================ STATIC FILE SERVING ================
const distPath = path.join(__dirname, 'dist')
app.use(express.static(distPath))
app.get('*', (_req, res) => {
  res.sendFile(path.join(distPath, 'index.html'))
})

const PORT = process.env.PORT || 3000
app.listen(PORT, '0.0.0.0', () => {
  console.log(`QBench Dashboard v2 running on http://0.0.0.0:${PORT}`)
  console.log(`Database: ${DB_PATH}`)
})
