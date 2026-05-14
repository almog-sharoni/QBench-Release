import { useState, useEffect } from 'react'
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import { LayoutDashboard, FlaskConical, HardDrive, Zap, BarChart3, GitGraph, Sun, Moon } from 'lucide-react'
import DashboardHome from './pages/DashboardHome'
import Experiments from './pages/Experiments'
import RunModels from './pages/RunModels'
import CacheSimulation from './pages/CacheSimulation'
import GraphViewer from './pages/GraphViewer'

const NAV = [
  { to: '/', icon: BarChart3, label: 'Overview' },
  { to: '/experiments', icon: FlaskConical, label: 'Experiments' },
  { to: '/cache', icon: HardDrive, label: 'Cache Simulation' },
  { to: '/graph', icon: GitGraph, label: 'Architecture Graph' },
  { to: '/run', icon: Zap, label: 'Run Models' },
]

export default function App() {
  const [dark, setDark] = useState(true)

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
  }, [dark])

  return (
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
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all duration-150 ${
                    isActive
                      ? 'bg-blue-500/10 text-blue-400 font-medium border border-blue-500/20'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                  }`
                }
              >
                <Icon size={17} />
                {label}
              </NavLink>
            ))}
          </nav>

          <div className="p-3 border-t border-slate-700/30">
            <button
              onClick={() => setDark(!dark)}
              className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 transition-colors"
            >
              {dark ? <Sun size={16} /> : <Moon size={16} />}
              {dark ? 'Light mode' : 'Dark mode'}
            </button>
          </div>
        </aside>

        <main className="flex-1 overflow-auto">
          <Routes>
            <Route path="/" element={<DashboardHome />} />
            <Route path="/experiments" element={<Experiments />} />
            <Route path="/cache" element={<CacheSimulation />} />
            <Route path="/graph" element={<GraphViewer />} />
            <Route path="/run" element={<RunModels />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
