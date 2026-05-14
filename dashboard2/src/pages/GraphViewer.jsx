import { useState } from 'react'
import { fetchModelGraphs } from '../lib/api'
import { useAPI } from '../hooks/useAPI'
import { GitGraph, RefreshCw, ZoomIn, Maximize2 } from 'lucide-react'

export default function GraphViewer() {
  const [selectedModel, setSelectedModel] = useState('')
  const [graphJson, setGraphJson] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const { data: graphList } = useAPI(() => fetchModelGraphs(), [])

  const models = graphList && graphList.length > 0
    ? [...new Set(graphList.map(g => g.model_name))]
    : ['resnet18', 'resnet50']

  const loadGraph = async () => {
    if (!selectedModel) return
    setLoading(true)
    setError(null)
    try {
      const data = await fetchModelGraphs(selectedModel)
      if (data && data.length > 0) {
        setGraphJson(data[0])
      } else {
        setError('No graph data found for this model')
        setGraphJson(null)
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-white flex items-center gap-2">
          <GitGraph size={24} className="text-green-400" />
          Architecture Graph
        </h2>
        <p className="text-slate-400 text-sm mt-1">Model quantization map visualization</p>
      </div>

      <div className="flex items-center gap-3">
        <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)}
          className="bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-2.5 text-sm text-slate-200 focus:outline-none focus:border-blue-500/50 min-w-[200px]">
          <option value="">Select a model...</option>
          {models.map(m => <option key={m} value={m}>{m}</option>)}
        </select>
        <button onClick={loadGraph} disabled={!selectedModel || loading}
          className="px-4 py-2.5 rounded-lg text-sm bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-40 transition-colors flex items-center gap-2">
          {loading ? <RefreshCw size={15} className="animate-spin" /> : <ZoomIn size={15} />}
          {loading ? 'Loading...' : 'Load Graph'}
        </button>
      </div>

      {error && (
        <div className="bg-red-500/10 text-red-400 border border-red-500/20 rounded-xl px-4 py-3 text-sm">{error}</div>
      )}

      {graphJson && (
        <div className="glass rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-semibold text-white">{graphJson.model_name}</h3>
              <p className="text-xs text-slate-400">
                Generated: {graphJson.generated_at || 'unknown'}
                {graphJson.graph_size_original && ` · ${(graphJson.graph_size_original / 1024).toFixed(1)} KB`}
              </p>
            </div>
          </div>

          <div className="border border-slate-700/30 rounded-lg overflow-hidden" style={{ height: '600px' }}>
            <iframe
              srcDoc={renderGraphHtml(graphJson)}
              style={{ width: '100%', height: '100%', border: 'none' }}
              title="Architecture Graph"
              sandbox="allow-scripts"
            />
          </div>
        </div>
      )}

      {!graphJson && !loading && (
        <div className="glass rounded-xl p-8 text-center">
          <GitGraph size={48} className="mx-auto mb-4 text-slate-600" />
          <p className="text-slate-400">Select a model and load its architecture graph.</p>
          <p className="text-xs text-slate-500 mt-1">
            {graphList && graphList.length > 0
              ? `${graphList.length} graph(s) available in database`
              : 'Graphs are generated during evaluation. Run with graph generation enabled.'}
          </p>
        </div>
      )}
    </div>
  )
}

function renderGraphHtml(graphJson) {
  // Simple graph visualization using HTML canvas or SVG
  // For complex Cytoscape graphs, we'd need to pull in the library
  // This provides a basic tree view of the graph structure
  const jsonStr = JSON.stringify(graphJson, null, 2)

  return `<!DOCTYPE html>
<html>
<head>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { background: #0f172a; color: #e2e8f0; font-family: monospace; padding: 20px; overflow: auto; height: 100vh; }
    .node { margin-left: 20px; border-left: 1px solid rgba(148,163,184,0.2); padding-left: 12px; }
    .node-name { font-size: 13px; color: #94a3b8; padding: 4px 0; cursor: pointer; }
    .node-name:hover { color: #e2e8f0; }
    .quantized { color: #a7f3d0 !important; }
    .quantized::before { content: '● '; }
    .collapsed > .children { display: none; }
    button { background: rgba(59,130,246,0.15); border: 1px solid rgba(59,130,246,0.3); color: #60a5fa; padding: 2px 10px; border-radius: 4px; cursor: pointer; font-size: 11px; margin: 4px 0; }
    button:hover { background: rgba(59,130,246,0.25); }
    .fullscreen { position: fixed; top: 10px; right: 10px; z-index: 10; }
  </style>
</head>
<body>
  <button class="fullscreen" onclick="document.documentElement.requestFullscreen()">Fullscreen</button>
  <h3 style="margin-bottom:16px;color:#f1f5f9;font-size:14px;">Architecture Graph (${graphJson.model_name || 'Model'})</h3>
  <pre style="font-size:11px;color:#94a3b8;max-height:100%;overflow:auto;">${jsonStr.replace(/</g, '&lt;')}</pre>
</body>
</html>`
}
