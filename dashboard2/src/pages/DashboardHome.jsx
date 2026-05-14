import { fetchStats } from '../lib/api'
import { useAPI } from '../hooks/useAPI'
import { useMemo } from 'react'
import { FlaskConical, CheckCircle2, Cpu, TrendingUp, TrendingDown } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'

const COLORS = ['#3b82f6', '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b']

export default function DashboardHome() {
  const { data, loading } = useAPI(() => fetchStats(), [])

  const chartData = useMemo(() => {
    if (!data?.model_stats) return []
    return data.model_stats.map((m, i) => ({
      name: m.model_name.replace(/_/g, ' '),
      Accuracy: parseFloat(m.best_acc || 0),
      Reference: parseFloat(m.ref_acc1 || 0),
      delta: parseFloat(m.delta || 0),
      color: COLORS[i % COLORS.length],
    }))
  }, [data])

  if (loading) {
    return (
      <div className="p-8">
        <div className="space-y-4">
          {[1, 2, 3].map(i => (
            <div key={i} className="animate-shimmer rounded-xl h-24" />
          ))}
        </div>
      </div>
    )
  }

  if (!data) return null

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-white">Dashboard Overview</h2>
        <p className="text-slate-400 text-sm mt-1">Quantization benchmark results at a glance</p>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard icon={FlaskConical} label="Total Runs" value={data.total} color="blue" />
        <StatCard icon={CheckCircle2} label="Successful" value={data.success} color="emerald" />
        <StatCard icon={Cpu} label="Models" value={data.models} color="purple" />
        <StatCard icon={TrendingUp} label="Experiments" value={data.experiments} color="amber" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="glass rounded-xl p-5">
          <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
            <TrendingUp size={18} className="text-blue-400" />
            Model Accuracy Comparison
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData} margin={{ top: 5, right: 5, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(100,116,139,0.15)" />
              <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 12 }} angle={-20} textAnchor="end" height={60} />
              <YAxis domain={[60, 85]} tick={{ fill: '#94a3b8', fontSize: 12 }} />
              <Tooltip
                contentStyle={{ background: '#1e293b', border: '1px solid rgba(100,116,139,0.2)', borderRadius: 8, fontSize: 12 }}
                labelStyle={{ color: '#e2e8f0' }}
              />
              <Bar dataKey="Reference" fill="rgba(100,116,139,0.3)" radius={[4, 4, 0, 0]} name="FP32 Ref" />
              <Bar dataKey="Accuracy" radius={[4, 4, 0, 0]} name="Quantized">
                {chartData.map((entry, i) => (
                  <Cell key={i} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="glass rounded-xl p-5">
          <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
            <TrendingDown size={18} className="text-purple-400" />
            Accuracy Delta (Quantized vs FP32)
          </h3>
          <div className="space-y-3">
            {chartData.map((m, i) => (
              <div key={i} className="flex items-center gap-3">
                <span className="w-28 text-sm text-slate-300 truncate">{m.name}</span>
                <div className="flex-1 bg-slate-700/50 rounded-full h-2.5 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-red-500 to-amber-500"
                    style={{ width: `${Math.min(Math.abs(m.delta) * 40, 100)}%` }}
                  />
                </div>
                <span className={`text-xs font-mono w-16 text-right ${m.delta < 0 ? 'text-red-400' : 'text-green-400'}`}>
                  {m.delta > 0 ? '+' : ''}{m.delta}%
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

function StatCard({ icon: Icon, label, value, color }) {
  const colors = {
    blue: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
    emerald: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
    purple: 'bg-purple-500/10 text-purple-400 border-purple-500/20',
    amber: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
  }

  return (
    <div className="glass rounded-xl p-4 flex items-center gap-4">
      <div className={`w-10 h-10 rounded-lg flex items-center justify-center border ${colors[color]}`}>
        <Icon size={20} />
      </div>
      <div>
        <p className="text-2xl font-bold text-white tabular-nums">{value}</p>
        <p className="text-xs text-slate-400">{label}</p>
      </div>
    </div>
  )
}
