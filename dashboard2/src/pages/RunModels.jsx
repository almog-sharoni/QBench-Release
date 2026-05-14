import { useState, useEffect, useRef, useCallback } from 'react'
import { fetchModels, fetchConfigs, fetchRunStatus, launchRun, fetchRunLog } from '../lib/api'
import { useAPI } from '../hooks/useAPI'
import {
  Play, Zap, Cpu, Layers, Database, Settings, Terminal, RefreshCw,
  Circle, XCircle, CheckCircle2, Clock, FileText, StopCircle
} from 'lucide-react'

const DATASETS = [
  { name: 'imagenet', label: 'ImageNet', path: '/data/imagenet/val' },
  { name: 'imagenette', label: 'Imagenette', path: 'tests/data/imagenette2-320/val' },
]
const WEIGHT_FORMATS = ['fp32', 'fp16', 'bf16', 'fp8_e4m3', 'fp8_e5m2', 'fp6_e2m3', 'fp4_e2m1', 'fp4_e3m0', 'int8', 'int4']
const INPUT_FORMATS = ['fp32', 'fp16', 'bf16', 'fp8_e4m3', 'fp8_e5m2', 'dyn_input_mse', 'int8']
const EXPERIMENT_TYPES = [
  { value: 'fp32_ref', label: 'FP32 Reference', desc: 'Baseline without quantization' },
  { value: 'weight_quant', label: 'Weight Quantization', desc: 'Quantize weights only' },
  { value: 'input_quant', label: 'Input Quantization', desc: 'Quantize activations only' },
  { value: 'hybrid_quant', label: 'Hybrid Quantization', desc: 'Quantize weights + inputs' },
]

export default function RunModels() {
  const { data: modelsList } = useAPI(() => fetchModels(), [])
  const { data: configsList } = useAPI(() => fetchConfigs(), [])
  const models = modelsList || ['resnet18', 'resnet50']

  const [selectedModels, setSelectedModels] = useState([])
  const [dataset, setDataset] = useState('imagenet')
  const [weightFormat, setWeightFormat] = useState('fp8_e4m3')
  const [inputFormat, setInputFormat] = useState('dyn_input_mse')
  const [experimentType, setExperimentType] = useState('hybrid_quant')
  const [batchSize, setBatchSize] = useState(256)
  const [numBatches, setNumBatches] = useState(10)
  const [numWorkers, setNumWorkers] = useState(32)
  const [baseConfig, setBaseConfig] = useState('')
  const [messages, setMessages] = useState([])

  // Runner state
  const { data: runnerState, reload: reloadStatus } = useAPI(() => fetchRunStatus(), [])
  const [logContent, setLogContent] = useState('')
  const [logName, setLogName] = useState('')
  const [logSize, setLogSize] = useState(0)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const logInterval = useRef(null)

  const runningProcess = runnerState?.find(r => r.status === 'running')
  const lastFinished = runnerState?.find(r => r.status === 'finished')

  const toggleModel = (m) => {
    setSelectedModels(prev => prev.includes(m) ? prev.filter(x => x !== m) : [...prev, m])
  }

  const selectedDataset = DATASETS.find(d => d.name === dataset)

  const generateCommand = useCallback(() => {
    if (selectedModels.length === 0) return ''
    const components = ['python', '-m', 'runspace.core.runner']
    if (baseConfig) components.push('--base-config', baseConfig)
    components.push(
      '--models', selectedModels.join(','),
      '--dataset', dataset,
      '--dataset-path', selectedDataset?.path || '',
      '--weight-dt', weightFormat,
      '--activation-dt', inputFormat,
      '--experiment-type', experimentType,
      '--batch-size', String(batchSize),
      '--num-batches', String(numBatches),
      '--workers', String(numWorkers),
    )
    return components.join(' \\\n  ')
  }, [selectedModels, baseConfig, dataset, weightFormat, inputFormat, experimentType, batchSize, numBatches, numWorkers])

  const handleLaunch = async () => {
    if (selectedModels.length === 0) return
    const cmd = generateCommand().replace(/ \\\n  /g, ' ')
    const name = `dashboard_${experimentType}_${selectedModels[0]}_${weightFormat}`.replace(/[^a-zA-Z0-9\-_]/g, '_')
    try {
      const result = await launchRun(cmd, name)
      addMsg(`Launched! PID: ${result.pid}`, 'success')
      reloadStatus()
      setLogName(path_basename(result.log_path))
    } catch (e) {
      addMsg(`Launch failed: ${e.message}`, 'error')
    }
  }

  const addMsg = (text, type = 'info') => {
    setMessages(prev => [{ text, type, time: new Date().toLocaleTimeString() }, ...prev].slice(0, 20))
  }

  useEffect(() => {
    if (autoRefresh) {
      const timer = setInterval(() => reloadStatus(), 3000)
      return () => clearInterval(timer)
    }
  }, [autoRefresh, reloadStatus])

  useEffect(() => {
    if (runningProcess?.log_path) {
      const name = runningProcess.log_path.split('/').pop()
      setLogName(name)
      loadLog(name, 'tail')
    }
  }, [runningProcess?.log_path])

  const loadLog = async (name, mode) => {
    try {
      const data = mode === 'tail' ? await fetchRunLog(name, 131072) : await fetchRunLog(name)
      setLogContent(data.content)
      setLogSize(data.size)
    } catch {}
  }

  useEffect(() => {
    if (logName && autoRefresh && runningProcess) {
      logInterval.current = setInterval(() => loadLog(logName, 'tail'), 3000)
      return () => clearInterval(logInterval.current)
    }
  }, [logName, autoRefresh, runningProcess])

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white flex items-center gap-2">
            <Zap size={24} className="text-amber-400" />
            Run Models
          </h2>
          <p className="text-slate-400 text-sm mt-1">Launch quantization experiments</p>
        </div>
        <button onClick={() => reloadStatus()} className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-700/50">
          <RefreshCw size={16} />
        </button>
      </div>

      {messages.length > 0 && (
        <div className="space-y-1">
          {messages.slice(0, 3).map((m, i) => (
            <div key={i} className={`text-xs px-3 py-2 rounded-lg ${
              m.type === 'error' ? 'bg-red-500/10 text-red-400' :
              m.type === 'success' ? 'bg-emerald-500/10 text-emerald-400' :
              'bg-slate-800/50 text-slate-400'
            }`}>
              <span className="text-slate-600">{m.time}</span> {m.text}
            </div>
          ))}
        </div>
      )}

      {runningProcess && (
        <div className="glass rounded-xl p-4 space-y-3">
          <div className="flex items-center gap-3">
            <div className="w-3 h-3 rounded-full bg-amber-400 animate-pulse" />
            <span className="text-sm font-medium text-amber-400">Running</span>
            <span className="text-xs text-slate-500">PID: {runningProcess.pid}</span>
            <span className="text-xs text-slate-500 ml-auto">{runningProcess.started?.slice(11, 19)}</span>
          </div>
          <div className="animate-shimmer rounded-lg h-2" />
        </div>
      )}

      {lastFinished && !runningProcess && (
        <div className="glass rounded-xl p-4">
          <div className="flex items-center gap-3">
            {lastFinished.exit_code === 0 ? (
              <CheckCircle2 size={18} className="text-emerald-400" />
            ) : (
              <XCircle size={18} className="text-red-400" />
            )}
            <span className="text-sm text-slate-300">Last run finished (exit: {lastFinished.exit_code})</span>
            <span className="text-xs text-slate-500">{lastFinished.started?.slice(0, 19)}</span>
          </div>
        </div>
      )}

      {/* Model Selection */}
      <div className="glass rounded-xl p-5">
        <h3 className="font-semibold text-white mb-3 flex items-center gap-2">
          <Cpu size={16} className="text-blue-400" />
          Models ({selectedModels.length} selected)
        </h3>
        <div className="flex flex-wrap gap-2 max-h-48 overflow-auto">
          {models.map(m => (
            <button key={m} onClick={() => toggleModel(m)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                selectedModels.includes(m)
                  ? 'bg-blue-500/15 text-blue-400 border border-blue-500/30'
                  : 'bg-slate-800/50 text-slate-400 border border-slate-700/30 hover:border-slate-600/50'
              }`}>
              {m}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="glass rounded-xl p-5">
          <h3 className="font-semibold text-white mb-3">Experiment Type</h3>
          <div className="space-y-2">
            {EXPERIMENT_TYPES.map(et => (
              <button key={et.value} onClick={() => setExperimentType(et.value)}
                className={`w-full text-left p-3 rounded-lg transition-all ${
                  experimentType === et.value ? 'bg-purple-500/10 border border-purple-500/30' : 'bg-slate-800/30 border border-slate-700/20'
                }`}>
                <p className={`text-sm font-medium ${experimentType === et.value ? 'text-purple-400' : 'text-slate-300'}`}>{et.label}</p>
                <p className="text-xs text-slate-500 mt-0.5">{et.desc}</p>
              </button>
            ))}
          </div>
        </div>

        <div className="glass rounded-xl p-5">
          <h3 className="font-semibold text-white mb-3">Dataset</h3>
          <div className="space-y-2">
            {DATASETS.map(ds => (
              <button key={ds.name} onClick={() => setDataset(ds.name)}
                className={`w-full text-left p-3 rounded-lg transition-all ${
                  dataset === ds.name ? 'bg-emerald-500/10 border border-emerald-500/30' : 'bg-slate-800/30 border border-slate-700/20'
                }`}>
                <p className={`text-sm font-medium ${dataset === ds.name ? 'text-emerald-400' : 'text-slate-300'}`}>{ds.label}</p>
                <p className="text-xs text-slate-500 mt-0.5 font-mono">{ds.path}</p>
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="glass rounded-xl p-5">
          <h3 className="font-semibold text-white mb-3">Weight Format</h3>
          <select value={weightFormat} onChange={e => setWeightFormat(e.target.value)}
            className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-2.5 text-sm text-slate-200 focus:outline-none focus:border-blue-500/50">
            {WEIGHT_FORMATS.map(f => <option key={f} value={f}>{f}</option>)}
          </select>
        </div>
        <div className="glass rounded-xl p-5">
          <h3 className="font-semibold text-white mb-3">Input Format</h3>
          <select value={inputFormat} onChange={e => setInputFormat(e.target.value)}
            className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-2.5 text-sm text-slate-200 focus:outline-none focus:border-blue-500/50">
            {INPUT_FORMATS.map(f => <option key={f} value={f}>{f}</option>)}
          </select>
        </div>
      </div>

      <div className="glass rounded-xl p-5 space-y-3">
        <h3 className="font-semibold text-white flex items-center gap-2">
          <Settings size={16} className="text-slate-400" />
          Advanced Settings
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <NumField label="Batch Size" value={batchSize} onChange={setBatchSize} min={1} max={1024} />
          <NumField label="Num Batches" value={numBatches} onChange={setNumBatches} min={1} max={50000} />
          <NumField label="Workers" value={numWorkers} onChange={setNumWorkers} min={0} max={256} />
          <div>
            <label className="text-[10px] text-slate-400 uppercase tracking-wider">Base Config</label>
            <select value={baseConfig} onChange={e => setBaseConfig(e.target.value)}
              className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg px-2.5 py-2.5 mt-1 text-xs text-slate-200 focus:outline-none focus:border-blue-500/50">
              <option value="">None</option>
              {(configsList || []).map(c => <option key={c.name} value={c.name}>{c.name}</option>)}
            </select>
          </div>
        </div>
      </div>

      {generateCommand() && (
        <div className="glass rounded-xl p-5">
          <h3 className="font-semibold text-white mb-3 flex items-center gap-2">
            <Terminal size={16} className="text-slate-400" />
            Command Preview
          </h3>
          <pre className="bg-slate-900/80 rounded-lg p-4 text-xs text-slate-300 overflow-auto font-mono whitespace-pre-wrap">
            {generateCommand()}
          </pre>
        </div>
      )}

      <button onClick={handleLaunch} disabled={selectedModels.length === 0 || !!runningProcess}
        className={`w-full py-3 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition-all ${
          runningProcess ? 'bg-slate-800/50 text-slate-600 cursor-not-allowed' :
          selectedModels.length > 0 ? 'bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-500/20' :
          'bg-slate-800/50 text-slate-600 cursor-not-allowed'
        }`}>
        <Play size={16} />
        {runningProcess ? 'Run in progress...' :
         selectedModels.length > 0 ? `Run ${selectedModels.length} model(s)` : 'Select models'}
      </button>

      {logContent && (
        <div className="glass rounded-xl p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-white flex items-center gap-2">
              <FileText size={16} className="text-slate-400" />
              Run Log {logSize ? `(${(logSize/1024).toFixed(0)} KB)` : ''}
            </h3>
            <div className="flex items-center gap-2">
              <label className="flex items-center gap-1.5 text-xs text-slate-400 cursor-pointer">
                <input type="checkbox" checked={autoRefresh} onChange={e => setAutoRefresh(e.target.checked)}
                  className="rounded bg-slate-700 border-slate-600" />
                Auto-refresh
              </label>
              <button onClick={() => loadLog(logName, 'tail')} className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-700/50">
                <RefreshCw size={13} />
              </button>
            </div>
          </div>
          <pre className="bg-slate-950/80 rounded-lg p-4 text-xs text-slate-300 overflow-auto font-mono whitespace-pre-wrap max-h-96">
            {logContent.slice(-65536)}
          </pre>
        </div>
      )}
    </div>
  )
}

function NumField({ label, value, onChange, min, max }) {
  return (
    <div>
      <label className="text-[10px] text-slate-400 uppercase tracking-wider">{label}</label>
      <input type="number" value={value} onChange={e => onChange(parseInt(e.target.value) || 0)}
        min={min} max={max}
        className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-2.5 mt-1 text-sm text-white focus:outline-none focus:border-blue-500/50" />
    </div>
  )
}

function path_basename(p) {
  return p?.split('/').pop() || ''
}
