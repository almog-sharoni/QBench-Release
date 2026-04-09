# Model Graph Visualization & Database Storage

This feature allows you to generate, store, and visualize quantization graphs for all models in your QBench database. Graphs show which layers are quantized and include compression statistics.

## Overview

- **Database Storage**: Graphs are stored as **gzip-compressed SVG** (~70-90% size reduction)
- **Dashboard Integration**: View graphs directly in the Streamlit dashboard by selecting a model
- **Interactive Visualization**: Zoom, pan, and explore graphs with smooth controls
- **Compact**: Typical model graph: ~500KB original → ~50-100KB compressed
- **One-Time Generation**: Generate once, visualize anytime

## Quick Start

### 1. Generate Graphs for All Models

From the project root, within Apptainer (recommended):

```bash
# Within Apptainer (has all dependencies)
./run_apptainer.sh python3 runspace/src/database/generate_model_graphs.py

# Or locally (requires torch installed)
python3 runspace/src/database/generate_model_graphs.py
```

### 2. View Graphs in Dashboard

1. Run the dashboard:
   ```bash
   ./run_dashboard.sh
   ```

2. Navigate to **Model Architecture & Quantization** section

3. Select a model from the dropdown

4. Click the expander to view the graph with:
   - **Interactive controls**: Zoom, pan, reset view
   - **Size metrics**: Original vs. compressed size, compression ratio
   - **Color legend**: Green (Quantized), Gold (Supported), Pink (Unsupported), Gray (Structural)
   - **Graph metadata**: Number of nodes, quantized layers, generation date

### Interactive Graph Controls

- **🔍+ / 🔍− Buttons**: Zoom in/out with fixed increment
- **Scroll Wheel**: Continuous zoom in/out
- **Click & Drag**: Pan across the graph
- **Double-click**: Reset to original view
- **Zoom Level Display**: Shows current zoom percentage
- **Right-click**: Download/save the SVG file

## Advanced Usage

### Batch Generate with Specific Models

```bash
# Generate for specific models only
./run_apptainer.sh python3 runspace/src/database/generate_model_graphs.py \
  --models resnet50 vit_b_16 mobilenet_v3_large

# Skip models that already have graphs (incremental)
./run_apptainer.sh python3 runspace/src/database/generate_model_graphs.py \
  --skip-existing

# Force regeneration even if graphs exist
./run_apptainer.sh python3 runspace/src/database/generate_model_graphs.py --force

# Use a specific database path
./run_apptainer.sh python3 runspace/src/database/generate_model_graphs.py \
  --db-path /path/to/your/database.db
```

## Understanding the Visualization

The graph shows your model's computational flow with layers colored by type:

| Color | Meaning |
|-------|---------|
| 🟢 **Green** | Quantized layers (using low-precision arithmetic) |
| 🟡 **Gold** | Supported layers (can be quantized but aren't in current config) |
| 🔴 **Pink** | Unsupported layers (no quantization support yet) |
| ⚫ **Gray** | Structural/Other operations (activations, data movement, etc.) |

## Database Schema

New table added to `runs.db`:

```sql
CREATE TABLE model_graphs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT UNIQUE NOT NULL,
    svg_compressed BLOB,              -- gzip-compressed SVG content
    svg_size_original INTEGER,        -- Original SVG size in bytes
    svg_size_compressed INTEGER,      -- Compressed SVG size in bytes
    metadata TEXT,                    -- JSON with layer counts, etc.
    generated_date TEXT,              -- When the graph was generated
    status TEXT                       -- SUCCESS or FAILED
)
```

## Storage Optimization

### Compression Strategy

- **Algorithm**: gzip with maximum compression level (9)
- **Typical Reduction**: 70-90% size reduction
- **Example**: 
  - Original SVG: 500 KB
  - Compressed: 50-100 KB
  - All models (~50): ~5 MB total (instead of 25 MB)

### Database Footprint

```
Single model graph:     ~50-100 KB (compressed)
50 models:             ~2.5-5 MB
Plus quantization runs: ~1-2 MB per 1000 runs

Total for full QBench: ~5-10 MB (very compact)
```

## API Usage

### Python API

```python
from runspace.src.database.handler import RunDatabase

db = RunDatabase()

# Store a graph (from generate_model_graphs.py)
with open('quantization_graph.svg', 'r') as f:
    svg_content = f.read()

db.store_model_graph(
    model_name='resnet50',
    svg_content=svg_content,
    metadata={
        'num_nodes': 234,
        'num_quantized_layers': 45,
        'framework': 'pytorch'
    }
)

# Retrieve a graph
svg, metadata = db.get_model_graph_svg('resnet50')
print(f"Graph size (original): {len(svg)} bytes")
print(f"Metadata: {metadata}")

# Get metadata without loading full SVG
meta = db.get_model_graph_metadata('resnet50')
print(f"Compressed size: {meta['svg_size_compressed']} bytes")

# List all models with graphs
graphs_df = db.get_all_model_graphs()
print(graphs_df[['model_name', 'svg_size_original', 'svg_size_compressed']])

# Check if model has a graph
has_graph = db.has_model_graph('resnet50')
```

## Performance Considerations

### Generation Time

- **Single model**: ~10-30 seconds (depends on model size)
- **50 models**: ~10-15 minutes
- **One-time cost**: Run once, then instant access via dashboard

### Dashboard Load Time

- **Retrieving metadata**: ~10 ms (no decompression)
- **Loading & displaying graph**: ~100-500 ms (decompress + render)
- **Dashboard responsiveness**: No slowdown, graphs load on-demand

## Troubleshooting

### Graph Generation Fails

**Error**: `Could not load {model_name}`
```bash
# Solution: Most torchvision models are supported. Check model name:
python -c "from torchvision import models; print(dir(models))"
```

**Error**: `torch not installed`
```bash
# Solution 1: Run in Apptainer (has all dependencies)
./run_apptainer.sh python3 runspace/src/database/generate_model_graphs.py

# Solution 2: Install locally
pip install torch torchvision
python3 runspace/src/database/generate_model_graphs.py
```

**Error**: `Warning: Failed to generate quantization graph`
```bash
# This usually means the model tracing failed
# Some models with complex control flow may not be traceable with torch.fx
# The script continues and logs the failure
```

### Graph Not Showing in Dashboard

1. **Graph wasn't generated**: Run generation script first
   ```bash
   python runspace/src/database/generate_model_graphs.py
   ```

2. **Wrong database path**: Ensure both generation and dashboard use same DB
   ```bash
   # Check where database is stored
   python3 -c "from runspace.src.database.handler import RunDatabase; db = RunDatabase(); print(db.db_path)"
   ```

3. **Browser caching**: Hard refresh (Ctrl+Shift+R or Cmd+Shift+R)

## Python API Examples

### Export Graphs to Disk

```python
from runspace.src.database.handler import RunDatabase

db = RunDatabase()
graphs = db.get_all_model_graphs()

for _, row in graphs.iterrows():
    model_name = row['model_name']
    svg, meta = db.get_model_graph_svg(model_name)
    
    with open(f'graphs/{model_name}.svg', 'w') as f:
        f.write(svg)
    
    print(f"Exported {model_name}")
```

### Get Compression Statistics

```python
from runspace.src.database.handler import RunDatabase

db = RunDatabase()
graphs = db.get_all_model_graphs()

total_original = graphs['svg_size_original'].sum()
total_compressed = graphs['svg_size_compressed'].sum()
reduction = 100 * (1 - total_compressed / total_original)

print(f"Total graphs: {len(graphs)}")
print(f"Space saved: {(total_original - total_compressed) / 1024:.1f} KB")
print(f"Compression ratio: {reduction:.1f}%")
```

## Integration with Dashboard

The graph visualization appears in each table's visualization section:

1. **Location**: "Model Architecture & Quantization" section
2. **Trigger**: Select a model from the dropdown
3. **Display**: 
   - Expandable graph viewer
   - Size metrics (original, compressed, reduction %)
   - Color legend
   - Metadata details (optional)

## Future Enhancements

Potential improvements:

- [x] Interactive graph rendering (zoom, pan) ✨ **Now available!**
- [ ] Export graphs as PNG/PDF
- [ ] Compare architectures side-by-side
- [ ] Highlight specific quantized layers by name
- [ ] Layer-wise statistics (ops count, memory, etc.)
- [ ] Time-series of architecture changes

## Interactive Graph Features

The dashboard includes built-in zoom and pan controls for exploring large model graphs:

**Zoom Controls:**
- **Buttons**: Use 🔍+ and 🔍− buttons in the top-right corner (0.2x increment per click)
- **Scroll Wheel**: Scroll to zoom in/out continuously (0.1x increment per scroll)
- **Zoom Display**: Real-time zoom level percentage displayed in controls

**Pan Controls:**
- **Click & Drag**: Click and hold on the graph, then drag to move it around
- **Smooth Movement**: Pan follows your cursor position
- **Double-click**: Instantly reset to original view (100%, position 0,0)

**Implementation Details:**
- Pure JavaScript implementation (no external dependencies)
- SVG transform-based rendering for smooth performance
- Zoom range: 50% to 500% of original size
- Functions dynamically created per graph to avoid conflicts

**Performance:**
- Smooth 60 FPS zoom/pan on most browsers
- Works with graphs up to 50+ MB uncompressed
- Zero lag due to client-side rendering

## See Also

- [graph_viz.py](../utils/graph_viz.py) - Visualization engine
- [handler.py](./handler.py) - Database API
- [dashboard.py](./dashboard.py) - Streamlit frontend
- [README.md](./README.md) - Database documentation
