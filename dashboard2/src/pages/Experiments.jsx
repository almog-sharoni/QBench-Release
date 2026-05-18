import { useState, useMemo, useCallback, useEffect, useContext } from 'react'
import { fetchRuns, fetchStats, fetchCompare, deleteRuns, renameExperiment, createDB, analyzeWinRates, fetchRun, loadPresets, savePresets } from '../lib/api'
import { useAPI } from '../hooks/useAPI'
import { DBContext } from '../App'
import {
  Search, ChevronUp, ChevronDown, ChevronLeft, ChevronRight, X, Eye,
  ArrowUpDown, BarChart3, Trash2, Edit3, Database, Zap, RefreshCw,
  Filter, Save, Download, Calendar, TrendingDown
} from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, Legend } from 'recharts'

const PAGE_SIZE = 20
const COLORS = ['#3b82f6', '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b', '#ef4444', '#ec4899', '#6366f1', '#14b8a6', '#f97316']

export default function Experiments() {
  const { dbName, runKind } = useContext(DBContext)

  const [filters, setFilters] = useState({
    model: '', status: '', experiment: '', weight_dt: '', activation_dt: '',
    w_bits: '', a_bits: '', min_date: '', max_date: '',
    sort: 'id', order: 'desc', newest_only: 'true'
  })
  const [page, setPage] = useState(1)
  const [selectedRows, setSelectedRows] = useState(new Set())
  const [viewMode, setViewMode] = useState('table')
  const [detailRun, setDetailRun] = useState(null)
  const [winRateData, setWinRateData] = useState(null)
  const [compareChartData, setCompareChartData] = useState(null)
  const [dialogState, setDialogState] = useState(null)
  const [message, setMessage] = useState(null)
  const [presets, setPresets] = useState({})
  const [presetName, setPresetName] = useState('')

  useEffect(() => { loadPresets().then(setPresets).catch(() => {}) }, [])

  const apiParams = useMemo(() => ({ ...filters, db: dbName, page, limit: PAGE_SIZE }), [filters, dbName, page])

  const { data: stats, loading: statsLoading } = useAPI(() => fetchStats(dbName), [dbName])
  const { data: runsData, loading: runsLoading, reload: reloadRuns } = useAPI(() => fetchRuns(apiParams), [apiParams])

  const totalPages = Math.max(1, Math.ceil((runsData?.total || 0) / PAGE_SIZE))

  const showMsg = (text, type = 'success') => {
    setMessage({ text, type })
    setTimeout(() => setMessage(null), 4000)
  }

  const toggleSort = useCallback((col) => {
    setFilters(f => ({ ...f, sort: col, order: f.sort === col && f.order === 'desc' ? 'asc' : 'desc' }))
  }, [])

  const updateFilter = (k, v) => { setFilters(f => ({ ...f, [k]: v })); setPage(1) }

  const sortIcon = (col) => {
    if (filters.sort !== col) return <ArrowUpDown size={11} className="opacity-30" />
    return filters.order === 'asc' ? <ChevronUp size={13} className="text-blue-400" /> : <ChevronDown size={13} className="text-blue-400" />
  }

  const toggleRow = (id) => setSelectedRows(prev => {
    const next = new Set(prev)
    next.has(id) ? next.delete(id) : next.add(id)
    return next
  })

  const toggleAll = () => {
    if (!runsData?.rows) return
    setSelectedRows(prev => prev.size === runsData.rows.length ? new Set() : new Set(runsData.rows.map(r => r.id)))
  }

  const selectedIds = [...selectedRows]
  const selectedRunObjects = runsData?.rows?.filter(r => selectedRows.has(r.id)) || []

  const handleDelete = async () => {
    if (!selectedIds.length) return
    try { await deleteRuns(selectedIds); showMsg(`Deleted ${selectedIds.length} runs`); setSelectedRows(new Set()); reloadRuns() }
    catch (e) { showMsg(e.message, 'error') }
  }

  const handleRename = async (newType) => {
    try { await renameExperiment(selectedIds, newType); showMsg(`Updated to "${newType}"`); setDialogState(null); reloadRuns() }
    catch (e) { showMsg(e.message, 'error') }
  }

  const handleCreateDB = async (name) => {
    try { const r = await createDB(selectedIds, name); showMsg(`Created DB with ${r.rows} rows`); setDialogState(null) }
    catch (e) { showMsg(e.message, 'error') }
  }

  const handleSavePreset = async () => {
    if (!presetName) return
    const newPresets = { ...presets, [presetName]: { ...filters } }
    try { await savePresets(newPresets); setPresets(newPresets); showMsg('Preset saved'); setPresetName('') }
    catch (e) { showMsg(e.message, 'error') }
  }

  const loadPreset = (name) => {
    const p = presets[name]
    if (p) { setFilters(f => ({ ...f, ...p })); setPage(1) }
  }

  const deletePreset = async (name) => {
    const newPresets = { ...presets }
    delete newPresets[name]
    try { await savePresets(newPresets); setPresets(newPresets) } catch (e) { showMsg(e.message, 'error') }
  }

  const handleViewWinRates = async (runBase) => {
    let run = runBase
    if (!run.quant_map_json && !run.input_map_json) {
      try { run = await fetchRun(runBase.id) } catch { showMsg('Failed to load details', 'error'); return }
    }
    try {
      const maps = []
      if (run.quant_map_json) maps.push({ label: 'Weight Quant Map', data: run.quant_map_json })
      if (run.input_map_json) maps.push({ label: 'Input Quant Map', data: run.input_map_json })
      if (maps.length === 0) { showMsg('No quant map data', 'error'); return }
      const results = await Promise.all(maps.map(async m => ({ label: m.label, analysis: await analyzeWinRates(m.data) })))
      setWinRateData({ run, results })
    } catch (e) { showMsg(e.message, 'error') }
  }

  const handleCompareChart = () => {
    if (selectedRunObjects.length === 0) return
    const data = selectedRunObjects.map((r, i) => ({
      name: `${r.model_name.replace(/_/g, ' ')}\n${r.experiment_type?.slice(0, 12)}`,
      quant: parseFloat(r.acc1 || 0),
      ref: parseFloat((r._ref_acc1 ?? r.ref_acc1) || 0),
      delta: (parseFloat(r.acc1 || 0) - parseFloat((r._ref_acc1 ?? r.ref_acc1) || 0)).toFixed(2),
      color: COLORS[i % COLORS.length],
    }))
    setCompareChartData(data)
  }

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-3">
      {message && (
        <div className={`px-4 py-2.5 rounded-xl text-sm font-medium ${
          message.type === 'error' ? 'bg-red-500/10 text-red-400 border border-red-500/20' : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
        }`}>{message.text}</div>
      )}

      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">Experiments</h2>
          <p className="text-slate-400 text-sm mt-1">
            {runsData?.total || 0} runs across {stats?.models || 0} models
            {runKind === 'feature_matching' && <span className="text-purple-400 ml-2">(Feature Matching)</span>}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {Object.keys(presets).length > 0 && (
            <select onChange={e => e.target.value && loadPreset(e.target.value)} value=""
              className="bg-slate-800/50 border border-slate-700/50 rounded-lg px-2.5 py-1.5 text-xs text-slate-300">
              <option value="">Load preset...</option>
              {Object.keys(presets).map(k => <option key={k} value={k}>{k}</option>)}
            </select>
          )}
          <button onClick={() => setViewMode(viewMode === 'table' ? 'chart' : 'table')}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium flex items-center gap-1.5 ${
              viewMode === 'chart' ? 'bg-blue-500/10 text-blue-400 border border-blue-500/20' : 'text-slate-400 hover:text-slate-200'}`}>
            <BarChart3 size={14} />{viewMode === 'table' ? 'Chart' : 'Table'}
          </button>
        </div>
      </div>

      {viewMode === 'chart' ? (
        <CompareChartView data={compareChartData} selectedRunObjects={selectedRunObjects} onGenerate={handleCompareChart} />
      ) : (
        <>
          <FilterBar filters={filters} updateFilter={updateFilter} stats={stats} presets={presets}
            presetName={presetName} setPresetName={setPresetName} onSavePreset={handleSavePreset}
            onDeletePreset={deletePreset} />

          {selectedIds.length > 0 && (
            <div className="glass rounded-xl px-4 py-3 flex items-center gap-3 flex-wrap">
              <span className="text-sm text-emerald-400 font-medium">{selectedIds.length} selected</span>
              <div className="h-5 w-px bg-slate-700/50" />
              <ActionBtn icon={Zap} label="Win Rates" onClick={() => handleViewWinRates(selectedRunObjects[0])} disabled={selectedIds.length !== 1} />
              <ActionBtn icon={BarChart3} label="Compare Chart" onClick={handleCompareChart} />
              <ActionBtn icon={Edit3} label="Rename" onClick={() => setDialogState({ type: 'rename' })} />
              <ActionBtn icon={Database} label="Create DB" onClick={() => setDialogState({ type: 'createdb' })} />
              <ActionBtn icon={Save} label="Save Preset" onClick={() => { setPresetName(`preset_${Date.now()}`) }} />
              <ActionBtn icon={Trash2} label="Delete" onClick={handleDelete} danger />
              <button onClick={() => setSelectedRows(new Set())} className="ml-auto text-xs text-slate-500 hover:text-slate-300">Clear</button>
            </div>
          )}

          <DataTable
            runsData={runsData} runsLoading={runsLoading} filters={filters}
            toggleSort={toggleSort} sortIcon={sortIcon}
            selectedRows={selectedRows} toggleRow={toggleRow} toggleAll={toggleAll}
            setDetailRun={setDetailRun}
            page={page} setPage={setPage} totalPages={totalPages}
          />
        </>
      )}

      {detailRun && <RunDetailModal runId={detailRun.id} run={detailRun} onClose={() => setDetailRun(null)} onWinRates={() => handleViewWinRates(detailRun)} />}
      {winRateData && <WinRateModal data={winRateData} onClose={() => setWinRateData(null)} />}
      {compareChartData && <CompareChartModal data={compareChartData} onClose={() => setCompareChartData(null)} />}
      {dialogState?.type === 'rename' && <RenameDialog onConfirm={handleRename} onClose={() => setDialogState(null)} count={selectedIds.length} />}
      {dialogState?.type === 'createdb' && <CreateDBDialog onConfirm={handleCreateDB} onClose={() => setDialogState(null)} count={selectedIds.length} />}
    </div>
  )
}

function ActionBtn({ icon: Icon, label, onClick, danger, disabled }) {
  return (
    <button onClick={onClick} disabled={disabled}
      className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors disabled:opacity-30 ${
        danger ? 'text-red-400 hover:bg-red-500/10 border border-red-500/20' : 'text-slate-300 hover:bg-slate-700/50 border border-slate-700/30'
      }`}><Icon size={13} /> {label}</button>
  )
}

function FilterBar({ filters, updateFilter, stats, presets, presetName, setPresetName, onSavePreset, onDeletePreset }) {
  const models = stats?.models_list || []
  const exps = stats?.experiments_list || []
  const statuses = stats?.statuses_list || []
  const bits = stats?.available_bits || []

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        <div className="relative min-w-[150px]">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
          <input type="text" placeholder="Search models..." value={filters.model}
            onChange={e => updateFilter('model', e.target.value)}
            className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg pl-9 pr-3 py-2 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-blue-500/50" />
        </div>
        <MultiSelect label="Status" options={statuses} value={filters.status} onChange={v => updateFilter('status', v)} />
        <MultiSelect label="Experiment" options={exps} value={filters.experiment} onChange={v => updateFilter('experiment', v)} />
        <MultiSelect label="W Bits" options={bits.map(String)} value={filters.w_bits} onChange={v => updateFilter('w_bits', v)} />
        <MultiSelect label="A Bits" options={bits.map(String)} value={filters.a_bits} onChange={v => updateFilter('a_bits', v)} />

        <div className="flex items-center gap-1.5">
          <Calendar size={12} className="text-slate-500" />
          <input type="date" value={filters.min_date} onChange={e => updateFilter('min_date', e.target.value)}
            className="bg-slate-800/50 border border-slate-700/50 rounded px-2 py-1.5 text-[11px] text-slate-200" />
          <span className="text-slate-600 text-xs">to</span>
          <input type="date" value={filters.max_date} onChange={e => updateFilter('max_date', e.target.value)}
            className="bg-slate-800/50 border border-slate-700/50 rounded px-2 py-1.5 text-[11px] text-slate-200" />
        </div>

        <label className="flex items-center gap-1.5 text-[11px] text-slate-400 cursor-pointer select-none">
          <input type="checkbox" checked={filters.newest_only === 'true'}
            onChange={e => updateFilter('newest_only', e.target.checked ? 'true' : 'false')}
            className="rounded bg-slate-700 border-slate-600" /> Newest
        </label>
      </div>

      <div className="flex items-center gap-2">
        <input type="text" placeholder="Preset name..." value={presetName}
          onChange={e => setPresetName(e.target.value)}
          className="bg-slate-800/50 border border-slate-700/50 rounded px-2 py-1 text-[11px] text-slate-200 placeholder-slate-500 w-32" />
        <button onClick={onSavePreset} disabled={!presetName}
          className="text-[10px] px-2 py-1 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20 hover:bg-blue-500/20 disabled:opacity-30">Save Preset</button>
        {Object.keys(presets).map(k => (
          <span key={k} className="flex items-center gap-1 text-[10px] bg-slate-700/30 rounded px-2 py-0.5">
            <button onClick={() => { updateFilter('model', presets[k].model || ''); updateFilter('status', presets[k].status || ''); updateFilter('experiment', presets[k].experiment || ''); updateFilter('w_bits', presets[k].w_bits || ''); updateFilter('a_bits', presets[k].a_bits || ''); }} className="text-slate-400 hover:text-slate-200">{k}</button>
            <button onClick={() => onDeletePreset(k)} className="text-slate-600 hover:text-red-400"><X size={10} /></button>
          </span>
        ))}
        {Object.keys(filters).filter(k => filters[k] && filters[k] !== 'true' && k !== 'sort' && k !== 'order' && k !== 'newest_only').length > 0 && (
          <button onClick={() => setFilters(f => ({ ...f, model: '', status: '', experiment: '', weight_dt: '', activation_dt: '', w_bits: '', a_bits: '', min_date: '', max_date: '', sort: 'id', order: 'desc', newest_only: 'true' }))}
            className="text-[10px] text-slate-500 hover:text-red-400">Reset All</button>
        )}
      </div>
    </div>
  )
}

function MultiSelect({ label, options, value, onChange }) {
  const [open, setOpen] = useState(false)
  const selected = value ? value.split(',').filter(Boolean) : []

  const toggle = (opt) => {
    const next = selected.includes(opt) ? selected.filter(s => s !== opt) : [...selected, opt]
    onChange(next.join(','))
  }

  return (
    <div className="relative">
      <button onClick={() => setOpen(!open)}
        className="flex items-center gap-1 bg-slate-800/50 border border-slate-700/50 rounded-lg px-2.5 py-2 text-xs text-slate-300 hover:border-slate-600/50 whitespace-nowrap">
        <Filter size={11} />{label}{selected.length > 0 && <span className="text-blue-400">({selected.length})</span>}<ChevronDown size={11} />
      </button>
      {open && (
        <div className="absolute top-full mt-1 z-50 glass rounded-xl p-2 min-w-[180px] max-h-56 overflow-auto shadow-xl" onClick={e => e.stopPropagation()}>
          <div className="flex gap-1 mb-1">
            <button onClick={() => onChange('')} className="text-[10px] px-2 py-0.5 rounded bg-slate-700/50 text-slate-300">All</button>
            <button onClick={() => onChange('__none__')} className="text-[10px] px-2 py-0.5 rounded bg-slate-700/50 text-slate-300">None</button>
          </div>
          {options.map(opt => (
            <label key={opt} className="flex items-center gap-2 px-2 py-1.5 rounded hover:bg-slate-700/30 cursor-pointer text-xs">
              <input type="checkbox" checked={selected.includes(String(opt))} onChange={() => toggle(String(opt))} className="rounded bg-slate-700 border-slate-600" />
              <span className="text-slate-300">{String(opt)}</span>
            </label>
          ))}
          <button onClick={() => setOpen(false)} className="w-full mt-1 text-[10px] text-slate-500 hover:text-slate-300 py-1">Close</button>
        </div>
      )}
    </div>
  )
}

function DataTable({ runsData, runsLoading, filters, toggleSort, sortIcon, selectedRows, toggleRow, toggleAll, setDetailRun, page, setPage, totalPages }) {
  const allSelected = runsData?.rows?.length > 0 && selectedRows.size === runsData.rows.length

  return (
    <div className="glass rounded-xl overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-700/30 bg-slate-800/30">
              <th className="w-10 px-3 py-3"><input type="checkbox" checked={allSelected} onChange={toggleAll} className="rounded bg-slate-700 border-slate-600" /></th>
              <ColH onClick={() => toggleSort('model_name')}>{sortIcon('model_name')} Model</ColH>
              <ColH onClick={() => toggleSort('experiment_type')}>{sortIcon('experiment_type')} Type</ColH>
              <ColH>W DT</ColH>
              <ColH>A DT</ColH>
              <ColH onClick={() => toggleSort('acc1')}>{sortIcon('acc1')} Acc@1</ColH>
              <ColH>Ref</ColH>
              <ColH>Δ</ColH>
              <ColH onClick={() => toggleSort('mse')}>{sortIcon('mse')} MSE</ColH>
              <ColH>Status</ColH>
              <ColH onClick={() => toggleSort('run_date')}>{sortIcon('run_date')} Date</ColH>
              <ColH></ColH>
            </tr>
          </thead>
          <tbody>
            {runsLoading ? Array.from({ length: 5 }).map((_, i) => (
              <tr key={i} className="border-b border-slate-700/20">
                {Array.from({ length: 12 }).map((_, j) => <td key={j} className="px-3 py-3"><div className="h-4 animate-shimmer rounded w-12" /></td>)}
              </tr>
            )) : runsData?.rows?.length === 0 ? (
              <tr><td colSpan={12} className="px-4 py-12 text-center text-slate-500">No runs match filters</td></tr>
            ) : runsData?.rows?.map(run => {
              const ref = run._ref_acc1 ?? run.ref_acc1
              return (
                <tr key={run.id} className={`border-b border-slate-700/20 ${selectedRows.has(run.id) ? 'bg-blue-500/5' : 'hover:bg-slate-800/20'}`}>
                  <td className="px-3 py-2.5"><input type="checkbox" checked={selectedRows.has(run.id)} onChange={() => toggleRow(run.id)} className="rounded bg-slate-700 border-slate-600" /></td>
                  <td className="px-3 py-2.5 text-sm font-medium text-white">{run.model_name}</td>
                  <td className="px-3 py-2.5"><Badge text={run.experiment_type} /></td>
                  <td className="px-3 py-2.5"><code className="text-[10px] bg-slate-700/50 px-1.5 py-0.5 rounded text-slate-300">{run.weight_dt || '-'}</code></td>
                  <td className="px-3 py-2.5"><code className="text-[10px] bg-slate-700/50 px-1.5 py-0.5 rounded text-slate-300">{run.activation_dt || '-'}</code></td>
                  <td className="px-3 py-2.5 font-mono text-sm text-white">{run.acc1 != null ? run.acc1.toFixed(2) : '-'}</td>
                  <td className="px-3 py-2.5 font-mono text-sm text-slate-400">{ref != null ? ref.toFixed(2) : '-'}</td>
                  <td className="px-3 py-2.5 font-mono text-xs">
                    {run.acc1 != null && ref != null ? <span className={run.acc1 >= ref ? 'text-green-400' : 'text-red-400'}>{(run.acc1 - ref).toFixed(2)}</span> : '-'}
                  </td>
                  <td className="px-3 py-2.5 font-mono text-[10px] text-slate-400">{run.mse != null ? run.mse.toExponential(2) : '-'}</td>
                  <td className="px-3 py-2.5"><span className={`px-2 py-0.5 rounded-full text-[10px] font-medium ${run.status === 'SUCCESS' ? 'bg-emerald-500/15 text-emerald-400' : 'bg-red-500/15 text-red-400'}`}>{run.status}</span></td>
                  <td className="px-3 py-2.5 text-[10px] text-slate-400 whitespace-nowrap">{run.run_date?.slice(0, 10)}</td>
                  <td className="px-3 py-2.5"><button onClick={() => setDetailRun(run)} className="p-1.5 rounded-lg hover:bg-slate-700/50 text-slate-400 hover:text-white"><Eye size={14} /></button></td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <div className="flex items-center justify-between px-4 py-3 border-t border-slate-700/30">
        <span className="text-xs text-slate-500">{runsData ? `${((page-1)*PAGE_SIZE)+1}-${Math.min(page*PAGE_SIZE, runsData.total)} of ${runsData.total}` : ''}</span>
        <div className="flex items-center gap-1">
          <PageBtn onClick={() => setPage(p => Math.max(1, p-1))} disabled={page <= 1}><ChevronLeft size={15} /></PageBtn>
          <span className="text-xs text-slate-400 px-2">{page} / {totalPages}</span>
          <PageBtn onClick={() => setPage(p => Math.min(totalPages, p+1))} disabled={page >= totalPages}><ChevronRight size={15} /></PageBtn>
        </div>
      </div>
    </div>
  )
}

function ColH({ onClick, children }) {
  return <th onClick={onClick} className="px-3 py-3 text-left text-[10px] font-medium text-slate-400 uppercase tracking-wider cursor-pointer hover:text-slate-200 select-none whitespace-nowrap"><span className="flex items-center gap-1">{children}</span></th>
}

function Badge({ text }) {
  const c = !text ? 'bg-slate-500/20 text-slate-400' : text.includes('ref') ? 'bg-blue-500/15 text-blue-400' : text.includes('hybrid') ? 'bg-purple-500/15 text-purple-400' : text.includes('weight') ? 'bg-amber-500/15 text-amber-400' : text.includes('input') ? 'bg-emerald-500/15 text-emerald-400' : 'bg-slate-500/20 text-slate-400'
  return <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium ${c}`}>{text || '-'}</span>
}

function PageBtn({ onClick, disabled, children }) {
  return <button onClick={onClick} disabled={disabled} className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-700/50 disabled:opacity-30 disabled:cursor-not-allowed">{children}</button>
}

function CompareChartView({ data, selectedRunObjects, onGenerate }) {
  if (!data) {
    return (
      <div className="glass rounded-xl p-8 text-center">
        <BarChart3 size={40} className="mx-auto mb-3 text-slate-600" />
        <p className="text-slate-400 mb-3">Select rows and click "Compare Chart" to generate</p>
        <button onClick={onGenerate} disabled={!selectedRunObjects?.length}
          className="px-4 py-2 rounded-lg text-sm bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-30">Generate from {selectedRunObjects?.length || 0} selected</button>
      </div>
    )
  }
  return (
    <div className="glass rounded-xl p-5">
      <h3 className="font-semibold text-white mb-4">Accuracy Comparison</h3>
      <ResponsiveContainer width="100%" height={400}>
        <BarChart data={data} margin={{ top: 10, right: 10, bottom: 5, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(100,116,139,0.1)" />
          <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 10 }} angle={-15} textAnchor="end" height={80} />
          <YAxis domain={[60, 'auto']} tick={{ fill: '#94a3b8', fontSize: 12 }} />
          <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid rgba(100,116,139,0.2)', borderRadius: 8, fontSize: 12 }} />
          <Legend />
          <Bar dataKey="ref" fill="rgba(100,116,139,0.3)" radius={[4,4,0,0]} name="Ref Acc" />
          <Bar dataKey="quant" radius={[4,4,0,0]} name="Quantized">
            {data.map((e, i) => <Cell key={i} fill={e.color} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

function CompareChartModal({ data, onClose }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={onClose}>
      <div className="glass rounded-2xl w-full max-w-3xl max-h-[85vh] overflow-auto m-4 p-5" onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold text-white">Accuracy Comparison</h3>
          <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-slate-700/50 text-slate-400 hover:text-white"><X size={18} /></button>
        </div>
        <ResponsiveContainer width="100%" height={420}>
          <BarChart data={data} margin={{ top: 10, right: 10, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(100,116,139,0.1)" />
            <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 10 }} angle={-15} textAnchor="end" height={80} />
            <YAxis domain={[60, 'auto']} tick={{ fill: '#94a3b8', fontSize: 12 }} />
            <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid rgba(100,116,139,0.2)', borderRadius: 8, fontSize: 12 }} />
            <Legend />
            <Bar dataKey="ref" fill="rgba(100,116,139,0.3)" radius={[4,4,0,0]} name="Ref Acc" />
            <Bar dataKey="quant" radius={[4,4,0,0]} name="Quantized">
              {data.map((e, i) => <Cell key={i} fill={e.color} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function RunDetailModal({ runId, run, onClose, onWinRates }) {
  const [fullRun, setFullRun] = useState(run)
  const [loadingFull, setLoadingFull] = useState(true)

  useEffect(() => {
    if (run.config_json || run.quant_map_json) { setFullRun(run); setLoadingFull(false); return }
    fetchRun(runId).then(r => { setFullRun(r); setLoadingFull(false) }).catch(() => setLoadingFull(false))
  }, [runId])

  let config = null
  try { config = fullRun.config_json ? JSON.parse(fullRun.config_json) : null } catch {}

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={onClose}>
      <div className="glass rounded-2xl w-full max-w-2xl max-h-[85vh] overflow-auto m-4" onClick={e => e.stopPropagation()}>
        <div className="sticky top-0 flex items-center justify-between p-5 border-b border-slate-700/30 bg-slate-900/90 backdrop-blur z-10">
          <div><h3 className="font-semibold text-white">{fullRun.model_name}</h3><p className="text-xs text-slate-400">#{fullRun.id} · {fullRun.experiment_type} · {fullRun.run_date}</p></div>
          <div className="flex items-center gap-2">
            {(fullRun.quant_map_json || fullRun.input_map_json) && <button onClick={onWinRates} className="px-3 py-1.5 rounded-lg text-xs bg-purple-500/10 text-purple-400 border border-purple-500/20 hover:bg-purple-500/20"><Zap size={12} className="inline mr-1" />Win Rates</button>}
            <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-slate-700/50 text-slate-400 hover:text-white"><X size={18} /></button>
          </div>
        </div>
        <div className="p-5 space-y-4">
          {loadingFull ? <div className="animate-shimmer rounded-xl h-64" /> : <>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              <KV l="Weight" v={fullRun.weight_dt} /><KV l="Input" v={fullRun.activation_dt} /><KV l="Output" v={fullRun.output_dt} />
              <KV l="Acc@1" v={fullRun.acc1 != null ? `${fullRun.acc1.toFixed(2)}%` : '-'} />
              <KV l="Ref Acc" v={fullRun._ref_acc1 != null ? `${fullRun._ref_acc1.toFixed(2)}%` : fullRun.ref_acc1 != null ? `${fullRun.ref_acc1.toFixed(2)}%` : '-'} />
              <KV l="Acc@5" v={fullRun.acc5 != null ? `${fullRun.acc5.toFixed(2)}%` : '-'} />
              <KV l="MSE" v={fullRun.mse != null ? fullRun.mse.toExponential(4) : '-'} />
              <KV l="L1" v={fullRun.l1 != null ? fullRun.l1.toExponential(4) : '-'} />
              <KV l="Certainty" v={fullRun.certainty != null ? fullRun.certainty.toFixed(4) : '-'} />
              <KV l="Status" v={fullRun.status} /><KV l="Experiment" v={fullRun.experiment_type} />
            </div>
            {config && <><h4 className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">Configuration</h4>
              <pre className="bg-slate-800/50 rounded-lg p-3 text-xs text-slate-300 overflow-auto max-h-48 font-mono">{JSON.stringify(config, null, 2)}</pre></>}
            {fullRun.cli_command && <><h4 className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">CLI Command</h4>
              <pre className="bg-slate-800/50 rounded-lg p-3 text-xs text-slate-300 overflow-auto max-h-32 font-mono whitespace-pre-wrap break-all">{fullRun.cli_command}</pre></>}
          </>}
        </div>
      </div>
    </div>
  )
}

function KV({ l, v }) {
  return <div className="bg-slate-800/30 rounded-lg p-2.5"><p className="text-[10px] text-slate-500 uppercase">{l}</p><p className="text-xs text-slate-200 mt-0.5">{v || '-'}</p></div>
}

function WinRateModal({ data, onClose }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={onClose}>
      <div className="glass rounded-2xl w-full max-w-2xl max-h-[85vh] overflow-auto m-4" onClick={e => e.stopPropagation()}>
        <div className="sticky top-0 flex items-center justify-between p-5 border-b border-slate-700/30 bg-slate-900/90 backdrop-blur z-10">
          <div><h3 className="font-semibold text-white">Layer Quantization Breakdown</h3><p className="text-xs text-slate-400">{data.run.model_name} · {data.run.weight_dt} / {data.run.activation_dt}</p></div>
          <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-slate-700/50 text-slate-400 hover:text-white"><X size={18} /></button>
        </div>
        <div className="p-5 space-y-5">
          {data.results.map((result, idx) => (
            <div key={idx} className="space-y-3">
              <h4 className="font-medium text-white text-sm">{result.label}</h4>
              {result.analysis ? (
                <>
                  <div className="grid grid-cols-4 gap-3">
                    <div className="bg-slate-800/30 rounded-lg p-3 text-center"><p className="text-lg font-bold text-white">{result.analysis.meta.total_layers}</p><p className="text-[9px] text-slate-400 uppercase">Layers</p></div>
                    <div className="bg-slate-800/30 rounded-lg p-3 text-center"><p className="text-lg font-bold text-white">{result.analysis.meta.total_chunks}</p><p className="text-[9px] text-slate-400 uppercase">Chunks</p></div>
                    <div className="bg-slate-800/30 rounded-lg p-3 text-center"><p className="text-lg font-bold text-white">{result.analysis.meta.top_format}</p><p className="text-[9px] text-slate-400 uppercase">Top Format</p></div>
                    <div className="bg-slate-800/30 rounded-lg p-3 text-center"><p className="text-lg font-bold text-white">{result.analysis.meta.unique_formats}</p><p className="text-[9px] text-slate-400 uppercase">Formats</p></div>
                  </div>
                  <table className="w-full text-xs">
                    <thead><tr className="border-b border-slate-700/30"><th className="text-left py-2 text-slate-400">Format</th><th className="text-right py-2 text-slate-400">Layer Wins</th><th className="text-right py-2 text-slate-400">Win Rate</th><th className="text-right py-2 text-slate-400">Chunks</th><th className="text-right py-2 text-slate-400">Chunk Rate</th></tr></thead>
                    <tbody>{result.analysis.summary.map((s, i) => (
                      <tr key={i} className="border-b border-slate-700/20"><td className="py-2 font-mono text-slate-300">{s.format}</td><td className="py-2 text-right text-slate-300">{s.layer_wins}</td><td className="py-2 text-right text-slate-400">{s.layer_win_rate}%</td><td className="py-2 text-right text-slate-300">{s.chunk_wins}</td><td className="py-2 text-right text-slate-400">{s.chunk_win_rate}%</td></tr>
                    ))}</tbody>
                  </table>
                </>
              ) : <p className="text-slate-500 text-xs">No data</p>}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function RenameDialog({ onConfirm, onClose, count }) {
  const [value, setValue] = useState('')
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={onClose}>
      <div className="glass rounded-2xl w-full max-w-md p-6 m-4 space-y-4" onClick={e => e.stopPropagation()}>
        <h3 className="font-semibold text-white">Rename ({count} runs)</h3>
        <input type="text" placeholder="New experiment type..." value={value} onChange={e => setValue(e.target.value)}
          className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-2.5 text-sm text-white"
          onKeyDown={e => e.key === 'Enter' && value && onConfirm(value)} />
        <div className="flex gap-2 justify-end">
          <button onClick={onClose} className="px-4 py-2 rounded-lg text-xs text-slate-400">Cancel</button>
          <button onClick={() => onConfirm(value)} disabled={!value} className="px-4 py-2 rounded-lg text-xs bg-blue-600 text-white disabled:opacity-40">Save</button>
        </div>
      </div>
    </div>
  )
}

function CreateDBDialog({ onConfirm, onClose, count }) {
  const [value, setValue] = useState(`selected_${new Date().toISOString().slice(0,10).replace(/-/g,'')}.db`)
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={onClose}>
      <div className="glass rounded-2xl w-full max-w-md p-6 m-4 space-y-4" onClick={e => e.stopPropagation()}>
        <h3 className="font-semibold text-white">Create DB ({count} rows)</h3>
        <input type="text" value={value} onChange={e => setValue(e.target.value)}
          className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-2.5 text-sm text-white" />
        <div className="flex gap-2 justify-end">
          <button onClick={onClose} className="px-4 py-2 rounded-lg text-xs text-slate-400">Cancel</button>
          <button onClick={() => onConfirm(value)} disabled={!value} className="px-4 py-2 rounded-lg text-xs bg-emerald-600 text-white disabled:opacity-40">Create</button>
        </div>
      </div>
    </div>
  )
}
