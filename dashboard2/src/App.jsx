import { useState, useEffect } from 'react'
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import { createContext, useContext, useCallback } from 'react'
import { LayoutDashboard, FlaskConical, HardDrive, Zap, BarChart3, GitGraph, Sun, Moon, Database, Settings, RefreshCw, Trash2, Edit3 } from 'lucide-react'
import { listDBs, renameDB, deleteDB } from './lib/api'
import DashboardHome from './pages/DashboardHome'
import Experiments from './pages/Experiments'
import RunModels from './pages/RunModels'
import CacheSimulation from './pages/CacheSimulation'
import GraphViewer from './pages/GraphViewer'

export const DBContext = createContext({ dbPath: 'runs.db', runKind: 'classification' })

const NAV = [
  { to: '/', icon: BarChart3, label: 'Overview' },
  { to: '/experiments', icon: FlaskConical, label: 'Experiments' },
  { to: '/cache', icon: HardDrive, label: 'Cache Simulation' },
  { to: '/graph', icon: GitGraph, label: 'Architecture Graph' },
  { to: '/run', icon: Zap, label: 'Run Models' },
]

export default function App() {
  const [dark, setDark] = useState(true)
  const [dbName, setDbName] = useState('runs.db')
  const [dbList, setDbList] = useState([])
  const [runKind, setRunKind] = useState('classification')
  const [expandDb, setExpandDb] = useState(false)
  const [renameTarget, setRenameTarget] = useState(null)
  const [deleteTarget, setDeleteTarget] = useState(null)
  const [message, setMessage] = useState(null)

  useEffect(() => { document.documentElement.classList.toggle('dark', dark) }, [dark])

  const loadDBs = useCallback(async () => {
    try { setDbList(await listDBs()) } catch {}
  }, [])

  useEffect(() => { loadDBs() }, [loadDBs])

  const flash = (text, type = 'success') => {
    setMessage({ text, type })
    setTimeout(() => setMessage(null), 3000)
  }

  const handleRename = async () => {
    if (!renameTarget) return
    try {
      const newName = prompt('New name:', renameTarget.name)
      if (!newName || newName === renameTarget.name) return
      await renameDB(renameTarget.name, newName)
      flash('Database renamed')
      setRenameTarget(null)
      loadDBs()
    } catch (e) { flash(e.message, 'error') }
  }

  const handleDelete = async () => {
    if (!deleteTarget) return
    try {
      await deleteDB(deleteTarget.name)
      flash('Database deleted')
      if (dbName === deleteTarget.name) setDbName('runs.db')
      setDeleteTarget(null)
      loadDBs()
    } catch (e) { flash(e.message, 'error') }
  }

  return (
    <DBContext.Provider value={{ dbName, runKind, setDbName, setRunKind }}>
      <BrowserRouter>
        <div className="flex h-screen overflow-hidden">
          <aside className="w-56 border-r border-slate-700/50 glass flex flex-col shrink-0">
            <div className="p-4 flex items-center gap-3 border-b border-slate-700/30">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shrink-0">
                <LayoutDashboard size={18} className="text-white" />
              </div>
              <div>
                <h1 className="font-bold text-sm tracking-tight text-white">QBench</h1>
                <p className="text-[10px] text-slate-400 uppercase tracking-wider">Dashboard v2</p>
              </div>
            </div>

            <nav className="flex-1 py-3 px-2 space-y-0.5 overflow-auto">
              {NAV.map(({ to, icon: Icon, label }) => (
                <NavLink key={to} to={to} end={to === '/'}
                  className={({ isActive }) =>
                    `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all duration-150 ${
                      isActive ? 'bg-blue-500/10 text-blue-400 font-medium border border-blue-500/20'
                               : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                    }`}>
                  <Icon size={17} />{label}
                </NavLink>
              ))}

              <div className="pt-3 mt-3 border-t border-slate-700/30">
                <button onClick={() => setExpandDb(!expandDb)}
                  className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 transition-colors">
                  <Database size={15} /> Database
                  <span className="ml-auto text-[10px] text-slate-600">{expandDb ? '▲' : '▼'}</span>
                </button>
                {expandDb && (
                  <div className="mt-1 ml-2 pl-3 border-l border-slate-700/30 space-y-1">
                    <div className="flex items-center gap-1 mb-2">
                      <span className="text-[9px] text-slate-500 uppercase">Run Type</span>
                      <select value={runKind} onChange={e => setRunKind(e.target.value)}
                        className="bg-slate-800 border border-slate-700/50 rounded px-1.5 py-0.5 text-[10px] text-slate-300 ml-auto">
                        <option value="classification">Classification</option>
                        <option value="feature_matching">Feature Matching</option>
                      </select>
                    </div>
                    <select value={dbName} onChange={e => setDbName(e.target.value)}
                      className="w-full bg-slate-800/50 border border-slate-700/50 rounded px-2 py-1.5 text-[11px] text-slate-200 mb-1">
                      {dbList.map(d => <option key={d.name} value={d.name}>{d.name}</option>)}
                    </select>
                    <div className="flex gap-1 flex-wrap">
                      {dbList.filter(d => d.name !== 'runs.db').map(d => (
                        <div key={d.name} className="flex items-center gap-0.5 text-[10px]">
                          <span className="text-slate-400 truncate max-w-[100px]">{d.name}</span>
                          <button onClick={() => setRenameTarget(d)} className="text-slate-600 hover:text-slate-300 p-0.5"><Edit3 size={10} /></button>
                          <button onClick={() => setDeleteTarget(d)} className="text-slate-600 hover:text-red-400 p-0.5"><Trash2 size={10} /></button>
                        </div>
                      ))}
                    </div>
                    <button onClick={loadDBs} className="text-[10px] text-slate-500 hover:text-slate-300 flex items-center gap-1">
                      <RefreshCw size={10} /> Refresh
                    </button>
                  </div>
                )}
              </div>
            </nav>

            <div className="p-3 border-t border-slate-700/30">
              <button onClick={() => setDark(!dark)}
                className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 transition-colors">
                {dark ? <Sun size={16} /> : <Moon size={16} />}{dark ? 'Light mode' : 'Dark mode'}
              </button>
            </div>
          </aside>

          <main className="flex-1 overflow-auto">
            {message && (
              <div className="fixed top-4 right-4 z-50 px-4 py-2.5 rounded-xl text-sm font-medium shadow-lg animate-in slide-in-from-right-4" style={{
                background: message.type === 'error' ? 'rgba(239,68,68,0.15)' : 'rgba(16,185,129,0.15)',
                color: message.type === 'error' ? '#fca5a5' : '#6ee7b7',
                border: `1px solid ${message.type === 'error' ? 'rgba(239,68,68,0.2)' : 'rgba(16,185,129,0.2)'}`,
              }}>{message.text}</div>
            )}
            <Routes>
              <Route path="/" element={<DashboardHome />} />
              <Route path="/experiments" element={<Experiments />} />
              <Route path="/cache" element={<CacheSimulation />} />
              <Route path="/graph" element={<GraphViewer />} />
              <Route path="/run" element={<RunModels />} />
            </Routes>
          </main>
        </div>

        {renameTarget && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={() => setRenameTarget(null)}>
            <div className="glass rounded-2xl p-6 m-4 max-w-md w-full space-y-4" onClick={e => e.stopPropagation()}>
              <h3 className="font-semibold text-white">Rename DB: {renameTarget.name}</h3>
              <input id="renameInput" className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-2.5 text-sm text-white" defaultValue={renameTarget.name} />
              <div className="flex gap-2 justify-end">
                <button onClick={() => setRenameTarget(null)} className="px-4 py-2 rounded-lg text-xs text-slate-400">Cancel</button>
                <button onClick={() => {
                  const v = document.getElementById('renameInput')?.value
                  if (v) { renameDB(renameTarget.name, v).then(() => { flash('Renamed'); setRenameTarget(null); loadDBs() }).catch(e => flash(e.message, 'error')) }
                }} className="px-4 py-2 rounded-lg text-xs bg-blue-600 text-white">Rename</button>
              </div>
            </div>
          </div>
        )}

        {deleteTarget && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={() => setDeleteTarget(null)}>
            <div className="glass rounded-2xl p-6 m-4 max-w-md w-full space-y-4" onClick={e => e.stopPropagation()}>
              <h3 className="font-semibold text-white">Delete DB?</h3>
              <p className="text-sm text-slate-400">Permanently delete <code className="text-red-400">{deleteTarget.name}</code>? This cannot be undone.</p>
              <div className="flex gap-2 justify-end">
                <button onClick={() => setDeleteTarget(null)} className="px-4 py-2 rounded-lg text-xs text-slate-400">Cancel</button>
                <button onClick={handleDelete} className="px-4 py-2 rounded-lg text-xs bg-red-600 text-white">Delete</button>
              </div>
            </div>
          </div>
        )}
      </BrowserRouter>
    </DBContext.Provider>
  )
}
