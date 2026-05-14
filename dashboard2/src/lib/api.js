const BASE = '/api'

async function fetchJSON(url, options = {}) {
  const res = await fetch(url, options)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

// Runs
export function fetchRuns(params = {}) {
  const q = new URLSearchParams()
  Object.entries(params).forEach(([k, v]) => { if (v !== undefined && v !== null && v !== '') q.set(k, v) })
  return fetchJSON(`${BASE}/runs?${q.toString()}`)
}

export function fetchRun(id) {
  return fetchJSON(`${BASE}/runs/${id}`)
}

export function deleteRuns(ids) {
  return fetchJSON(`${BASE}/runs`, { method: 'DELETE', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ids }) })
}

export function renameExperiment(ids, experiment_type) {
  return fetchJSON(`${BASE}/runs/experiment-type`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ids, experiment_type }) })
}

// Stats
export function fetchStats(db) {
  const q = db ? `?db=${encodeURIComponent(db)}` : ''
  return fetchJSON(`${BASE}/stats${q}`)
}

export function fetchCompare(db) {
  const q = db ? `?db=${encodeURIComponent(db)}` : ''
  return fetchJSON(`${BASE}/compare${q}`)
}

// DB Management
export function listDBs() {
  return fetchJSON(`${BASE}/db/list`)
}

export function renameDB(oldName, newName) {
  return fetchJSON(`${BASE}/db/rename`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ oldName, newName }) })
}

export function deleteDB(name) {
  return fetchJSON(`${BASE}/db/delete`, { method: 'DELETE', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name }) })
}

export function createDB(ids, name) {
  return fetchJSON(`${BASE}/db/create`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ids, name }) })
}

// Cache Simulations
export function fetchCacheSimulations(db) {
  const q = db ? `?db=${encodeURIComponent(db)}` : ''
  return fetchJSON(`${BASE}/cache-simulations${q}`)
}

// Model Graphs
export function fetchModelGraphs(model, db) {
  const params = []
  if (model) params.push(`model=${encodeURIComponent(model)}`)
  if (db) params.push(`db=${encodeURIComponent(db)}`)
  return fetchJSON(`${BASE}/model-graphs${params.length ? '?' + params.join('&') : ''}`)
}

// Feature Matching
export function fetchFMRuns(params = {}) {
  const q = new URLSearchParams()
  Object.entries(params).forEach(([k, v]) => { if (v !== undefined && v !== null && v !== '') q.set(k, v) })
  return fetchJSON(`${BASE}/fm-runs?${q.toString()}`)
}

// Configs & Models
export function fetchConfigs() { return fetchJSON(`${BASE}/configs`) }
export function fetchModels() { return fetchJSON(`${BASE}/models`) }

// Run Launcher
export function fetchRunStatus() { return fetchJSON(`${BASE}/run-launcher/status`) }
export function launchRun(command, name) {
  return fetchJSON(`${BASE}/run-launcher/launch`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ command, name }) })
}
export function fetchRunLog(name, tail) {
  const q = tail ? `?tail=${tail}` : ''
  return fetchJSON(`${BASE}/run-launcher/logs/${encodeURIComponent(name)}${q}`)
}

// Win Rate Analysis
export function analyzeWinRates(quant_map_json) {
  return fetchJSON(`${BASE}/analyze/win-rates`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ quant_map_json }) })
}
