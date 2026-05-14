import { useState, useEffect, useMemo } from 'react'
import { fetchCacheSimulations } from '../lib/api'
import { useAPI } from '../hooks/useAPI'
import { HardDrive, Database, Layers, Cpu, ChevronLeft, ChevronRight, RefreshCw } from 'lucide-react'

export default function CacheSimulation() {
  const { data: sims, loading, reload } = useAPI(() => fetchCacheSimulations(), [])
  const [selectedModel, setSelectedModel] = useState('')
  const [layerIndex, setLayerIndex] = useState(0)

  const modelSims = (sims || []).filter(s => !selectedModel || s.model_name === selectedModel)
  const models = [...new Set((sims || []).map(s => s.model_name))]
  const currentSim = modelSims.length > 0 ? modelSims[0] : null

  useEffect(() => { setLayerIndex(0) }, [selectedModel, currentSim?.id])

  let layers = []
  let rules = null
  try { layers = currentSim?.layers_json ? JSON.parse(currentSim.layers_json) : [] } catch {}
  try { rules = currentSim?.rules_json ? JSON.parse(currentSim.rules_json) : null } catch {}

  const numBanks = currentSim?.num_banks || 16
  const bankSize = currentSim?.bank_size || 128
  const cacheSize = currentSim?.cache_size_elements || 0

  const onChip = layers.filter(l => l.stay_on_chip !== false).length
  const offChip = layers.filter(l => l.stay_on_chip === false).length
  const totalLayers = layers.length

  const currentLayer = layers[layerIndex]
  const bankStates = useMemoBankStates(layers, numBanks, bankSize)

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-white flex items-center gap-2">
          <HardDrive size={24} className="text-blue-400" />
          Cache Simulation
        </h2>
        <p className="text-slate-400 text-sm mt-1">ASIC on-chip memory placement analysis</p>
      </div>

      {loading ? (
        <div className="animate-shimmer rounded-xl h-64" />
      ) : (sims || []).length === 0 ? (
        <div className="glass rounded-xl p-8 text-center">
          <Database size={40} className="mx-auto mb-3 text-slate-600" />
          <p className="text-slate-400">No cache simulation data available.</p>
          <p className="text-xs text-slate-500 mt-1">Run simulate_cache.py to generate cache simulation results.</p>
        </div>
      ) : (
        <>
          <div className="flex items-center gap-3 flex-wrap">
            <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)}
              className="bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500/50">
              <option value="">All Models</option>
              {models.map(m => <option key={m} value={m}>{m}</option>)}
            </select>
            <button onClick={reload} className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-700/50 transition-colors">
              <RefreshCw size={16} />
            </button>
          </div>

          {currentSim && (
            <>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <Stat label="Total Layers" value={totalLayers} icon={Layers} color="blue" />
                <Stat label="On Chip" value={onChip} icon={Cpu} color="emerald" />
                <Stat label="Off Chip (Quant)" value={offChip} icon={Database} color="amber" />
                <Stat label="Cache Banks" value={`${numBanks} x ${bankSize}`} icon={HardDrive} color="purple" />
              </div>

              {layers.length > 0 && (
                <BankVisualizer
                  layers={layers} currentIndex={layerIndex} setCurrentIndex={setLayerIndex}
                  numBanks={numBanks} bankSize={bankSize} cacheSize={cacheSize}
                  bankStates={bankStates}
                />
              )}

              <div className="glass rounded-xl p-5">
                <h3 className="font-semibold text-white mb-3">Layer Details</h3>
                <div className="overflow-x-auto max-h-96 overflow-y-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="sticky top-0 bg-slate-800/80 border-b border-slate-700/30">
                        <th className="text-left py-2 px-2 text-slate-400 font-medium">Layer</th>
                        <th className="text-left py-2 px-2 text-slate-400 font-medium">Type</th>
                        <th className="text-center py-2 px-2 text-slate-400 font-medium">On Chip</th>
                        <th className="text-left py-2 px-2 text-slate-400 font-medium">Rule</th>
                        <th className="text-right py-2 px-2 text-slate-400 font-medium">Input (K)</th>
                        <th className="text-right py-2 px-2 text-slate-400 font-medium">Output (K)</th>
                        <th className="text-right py-2 px-2 text-slate-400 font-medium">Weights (K)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {layers.map((l, i) => (
                        <tr key={i}
                          onClick={() => setLayerIndex(i)}
                          className={`border-b border-slate-700/20 cursor-pointer transition-colors ${i === layerIndex ? 'bg-blue-500/10' : 'hover:bg-slate-800/30'}`}>
                          <td className="py-2 px-2 text-slate-300">{l.name || `Layer ${i}`}</td>
                          <td className="py-2 px-2 text-slate-400">{l.type || '-'}</td>
                          <td className="py-2 px-2 text-center">
                            <span className={l.stay_on_chip !== false ? 'text-emerald-400' : 'text-red-400'}>
                              {l.stay_on_chip !== false ? 'Yes' : 'No'}
                            </span>
                          </td>
                          <td className="py-2 px-2 text-slate-400 font-mono text-[10px]">{l.rule || '-'}</td>
                          <td className="py-2 px-2 text-right text-slate-300 font-mono">{fmtK(l.input_elems)}</td>
                          <td className="py-2 px-2 text-right text-slate-300 font-mono">{fmtK(l.output_elems)}</td>
                          <td className="py-2 px-2 text-right text-slate-300 font-mono">{fmtK(l.weight_elems)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {rules && (
                <div className="glass rounded-xl p-5">
                  <h3 className="font-semibold text-white mb-3">Cache Rules Reference</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="border-b border-slate-700/30">
                          <th className="text-left py-2 px-2 text-slate-400 font-medium">Rule</th>
                          <th className="text-left py-2 px-2 text-slate-400 font-medium">On Chip</th>
                          <th className="text-left py-2 px-2 text-slate-400 font-medium">Xin Source</th>
                          <th className="text-left py-2 px-2 text-slate-400 font-medium">Applies To</th>
                          <th className="text-left py-2 px-2 text-slate-400 font-medium">Condition</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Array.isArray(rules) ? rules.map((r, i) => (
                          <tr key={i} className="border-b border-slate-700/20">
                            <td className="py-2 px-2 font-mono text-slate-300">{r.rule || r.name || `Rule ${i}`}</td>
                            <td className="py-2 px-2">{r.xout_on_chip !== false ? <span className="text-emerald-400">Yes</span> : <span className="text-red-400">No</span>}</td>
                            <td className="py-2 px-2 text-slate-400">{r.xin_from_cache !== false ? 'Cache' : 'External'}</td>
                            <td className="py-2 px-2 text-slate-400">{r.applies_to || '-'}</td>
                            <td className="py-2 px-2 text-slate-400">{r.stay_condition || r.reason || '-'}</td>
                          </tr>
                        )) : (
                          Object.entries(rules).map(([key, r]) => (
                            <tr key={key} className="border-b border-slate-700/20">
                              <td className="py-2 px-2 font-mono text-slate-300">{key}</td>
                              <td className="py-2 px-2 text-slate-400">{typeof r === 'string' ? r : JSON.stringify(r).slice(0, 80)}</td>
                            </tr>
                          ))
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </>
          )}
        </>
      )}
    </div>
  )
}

function Stat({ label, value, icon: Icon, color }) {
  const colors = {
    blue: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
    emerald: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
    amber: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
    purple: 'bg-purple-500/10 text-purple-400 border-purple-500/20',
  }
  return (
    <div className="glass rounded-xl p-4 flex items-center gap-3">
      <div className={`w-10 h-10 rounded-lg flex items-center justify-center border ${colors[color]}`}>
        <Icon size={18} />
      </div>
      <div>
        <p className="text-xl font-bold text-white">{value}</p>
        <p className="text-[10px] text-slate-400 uppercase tracking-wider">{label}</p>
      </div>
    </div>
  )
}

function BankVisualizer({ layers, currentIndex, setCurrentIndex, numBanks, bankSize, cacheSize, bankStates }) {
  const state = bankStates[currentIndex]
  if (!state) return null

  return (
    <div className="glass rounded-xl p-5 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-white">Memory Banks — {numBanks} x {bankSize}</h3>
        <div className="flex items-center gap-2">
          <button onClick={() => setCurrentIndex(i => Math.max(0, i - 1))} disabled={currentIndex <= 0}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-700/50 disabled:opacity-30 transition-colors">
            <ChevronLeft size={16} />
          </button>
          <span className="text-xs text-slate-400">{currentIndex + 1} / {layers.length}</span>
          <button onClick={() => setCurrentIndex(i => Math.min(layers.length - 1, i + 1))} disabled={currentIndex >= layers.length - 1}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-700/50 disabled:opacity-30 transition-colors">
            <ChevronRight size={16} />
          </button>
        </div>
      </div>

      <div className="flex items-center gap-3 flex-wrap">
        <span className="text-sm font-medium text-white">{layers[currentIndex]?.name || `Layer ${currentIndex}`}</span>
        <span className="text-xs text-slate-400">{layers[currentIndex]?.type || ''}</span>
        {layers[currentIndex]?.stay_on_chip !== false ? (
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/15 text-emerald-400">ON CHIP</span>
        ) : (
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-red-500/15 text-red-400">OFF CHIP</span>
        )}
      </div>

      <div className="flex gap-1 flex-wrap">
        {state.banks.map((b, i) => (
          <div key={i} className={`w-8 h-8 rounded flex items-center justify-center text-[9px] font-medium transition-colors ${
            b === 'xin' ? 'bg-orange-500/40 text-orange-300' :
            b === 'xout' ? 'bg-emerald-500/40 text-emerald-300' :
            b === 'xr' ? 'bg-purple-500/40 text-purple-300' :
            b === 'weights' ? 'bg-blue-500/40 text-blue-300' :
            b === 'stream' ? 'bg-amber-500/40 text-amber-300' :
            'bg-slate-700/50 text-slate-600'
          }`} title={`Bank ${i}: ${b}`}>
            {i}
          </div>
        ))}
      </div>

      <Legend />
    </div>
  )
}

function Legend() {
  const items = [
    { color: 'bg-orange-500/40', label: 'Xin (Input)' },
    { color: 'bg-emerald-500/40', label: 'Xout (Output)' },
    { color: 'bg-purple-500/40', label: 'Xr (Residual)' },
    { color: 'bg-blue-500/40', label: 'Weights' },
    { color: 'bg-amber-500/40', label: 'Stream Buffer' },
    { color: 'bg-slate-700/50', label: 'Free' },
  ]
  return (
    <div className="flex gap-4 flex-wrap">
      {items.map(({ color, label }) => (
        <div key={label} className="flex items-center gap-1.5">
          <div className={`w-3 h-3 rounded ${color}`} />
          <span className="text-[10px] text-slate-400">{label}</span>
        </div>
      ))}
    </div>
  )
}

function fmtK(n) {
  if (n == null) return '-'
  return `${(n / 1000).toFixed(1)}K`
}

function useMemoBankStates(layers, numBanks, bankSize) {
  return useMemo(() => {
    return layers.map(l => {
      const ob = Math.min(Math.ceil((l.output_elems || 0) / bankSize), numBanks)
      const wb = Math.min(Math.ceil((l.weight_elems || 0) / bankSize), numBanks)
      const ib = Math.min(Math.ceil((l.input_elems || 0) / bankSize), numBanks)
      const stay = l.stay_on_chip !== false

      let banks = []
      if (stay) {
        for (let i = 0; i < Math.min(ib, numBanks); i++) banks.push('xin')
        for (let i = 0; i < Math.min(ob, numBanks - banks.length); i++) banks.push('xout')
        for (let i = 0; i < Math.min(2, numBanks - banks.length); i++) banks.push('xr')
        for (let i = 0; i < Math.min(wb, numBanks - banks.length); i++) banks.push('weights')
        for (let i = 0; i < Math.min(2, numBanks - banks.length); i++) banks.push('stream')
      } else {
        for (let i = 0; i < Math.min(2, numBanks); i++) banks.push('stream')
        for (let i = 0; i < Math.min(wb, numBanks - banks.length); i++) banks.push('weights')
      }
      while (banks.length < numBanks) banks.push('free')

      return { ...l, xin_b: ib, xout_b: ob, wt_b: wb, banks, stay }
    })
  }, [layers, numBanks, bankSize])
}
