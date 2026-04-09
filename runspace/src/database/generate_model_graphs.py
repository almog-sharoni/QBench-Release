#!/usr/bin/env python3
"""
Generate and store quantization graphs for all models in the database.

This script:
1. Loads available models from YAML config
2. Generates quantization graphs using graph_viz.py
3. Stores compressed SVGs in the database with metadata
4. Can be re-run to update graphs for specific models

Usage:
    python generate_model_graphs.py                    # Generate for all models
    python generate_model_graphs.py --models resnet18 vit_b_16  # Specific models
    python generate_model_graphs.py --skip-existing    # Skip models that already have graphs
"""

import os
import sys
import argparse
import yaml
import tempfile
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.src.database.handler import RunDatabase
from runspace.src.adapters.generic_adapter import GenericAdapter

# Try to import torch and visualization tools, but make them optional
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  Warning: torch not installed. Graph generation may fail.")

try:
    from runspace.src.utils.graph_viz import generate_quantization_graph
    GRAPH_VIZ_AVAILABLE = True
except Exception as e:
    GRAPH_VIZ_AVAILABLE = False
    print(f"⚠️  Warning: graph_viz not available: {e}")


def load_model_names(config_file=None):
    """Load model names from config file or use defaults."""
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            if isinstance(config, list):
                return [m.get('name') for m in config if 'name' in m]
    
    # Fallback: Common model names
    default_models = [
        'resnet18', 'resnet50', 'resnet152',
        'vit_b_16', 'vit_l_16',
        'efficientnet_b0', 'efficientnet_v2_l',
        'mobilenet_v3_large',
        # 'densenet121', 'densenet161',
        'inception_v3',
        'alexnet', 'vgg19_bn',
        'googlenet',
    ]
    return default_models

def get_model_from_name(model_name, quantized=True):
    """
    Load a model by name, optionally quantized.
    
    Args:
        model_name (str): Name of the model (e.g., 'resnet50')
        quantized (bool): Whether to load the quantized version.
        
    Returns:
        torch.nn.Module: Loaded model
    """
    if not TORCH_AVAILABLE:
        return None
    
    try:
        if quantized:
            print(f"  Loading quantized {model_name} via GenericAdapter...")
            # We quantize all supported layers to show them as Green in the graph
            adapter = GenericAdapter(
                model_name=model_name,
                quantized_ops=["all"]
            )
            model = adapter.build_model(quantized=True)
            return model
        else:
            # Try torchvision models
            from torchvision import models
            
            model_fn = getattr(models, model_name, None)
            if model_fn:
                print(f"  Loading vanilla {model_name} from torchvision...")
                model = model_fn(weights=None)
                model.eval()
                return model
    except Exception as e:
        print(f"  Error loading {model_name}: {e}")
    
    return None

def generate_graph_for_model(model_name, db, force=False, quantized=True):
    """
    Generate quantization graph for a single model.
    
    Args:
        model_name (str): Name of the model
        db (RunDatabase): Database instance
        force (bool): Force regeneration even if exists
        quantized (bool): Whether to generate graph for quantized model
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Check dependencies
    if not TORCH_AVAILABLE:
        print(f"✗ Skipping {model_name}: torch not available")
        return False
    
    if not GRAPH_VIZ_AVAILABLE:
        print(f"✗ Skipping {model_name}: graph_viz not available")
        return False
    
    # Check if already exists
    if not force and db.has_model_graph(model_name):
        print(f"✓ Graph already exists for {model_name}, skipping...")
        return True
    
    print(f"\n📊 Generating {'quantized' if quantized else 'vanilla'} graph for {model_name}...", end=" ", flush=True)
    
    try:
        # Load model
        model = get_model_from_name(model_name, quantized=quantized)
        if model is None:
            print(f"✗ Could not load {model_name}")
            return False
        
        model.eval()
        
        # Generate SVG in temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as f:
            temp_svg = f.name
        
        try:
            # Generate graph visualization
            generate_quantization_graph(model, temp_svg, model_name=model_name)
            
            # Read generated SVG
            with open(temp_svg, 'r') as f:
                svg_content = f.read()
            
            # Extract metadata from the SVG (count nodes, quantized layers, etc.)
            num_nodes = svg_content.count('<title>')  # Rough estimate
            num_quantized = svg_content.count('#90EE90')  # Green color = quantized
            
            metadata = {
                'num_nodes': num_nodes,
                'num_quantized_layers': num_quantized,
                'generated_at': datetime.now().isoformat()
            }
            
            # Store in database
            db.store_model_graph(model_name, svg_content, metadata)
            print(f"✓ Success")
            return True
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_svg):
                os.remove(temp_svg)
                
    except Exception as e:
        print(f"✗ Error: {e}")
        # import traceback
        # traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--models', 
        nargs='+',
        help='Specific models to generate graphs for (default: all available)'
    )
    parser.add_argument(
        '--config',
        help='Path to model config YAML file'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip models that already have graphs in database'
    )
    parser.add_argument(
        '--db-path',
        help='Path to database file (default: runspace/database/runs.db)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Regenerate graphs even if they exist'
    )
    parser.add_argument(
        '--vanilla',
        action='store_true',
        help='Generate graph for unquantized (vanilla) model instead'
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not TORCH_AVAILABLE:
        print("❌ Error: torch is required for graph generation.")
        print("\nTo install torch in your Apptainer environment:")
        print("  1. Run: ./run_apptainer.sh pip install torch torchvision")
        print("  2. Or install locally: pip install torch torchvision")
        print("\nWithin Apptainer, torch should already be available.")
        return
    
    if not GRAPH_VIZ_AVAILABLE:
        print("❌ Error: graph_viz module not available.")
        print("Ensure graph_viz.py is in runspace/src/utils/ and all dependencies are installed.")
        return
    
    # Initialize database
    db = RunDatabase(db_path=args.db_path) if args.db_path else RunDatabase()
    
    # Get list of models to process
    if args.models:
        models_to_process = args.models
    else:
        models_to_process = load_model_names(args.config)
    
    print(f"🚀 Generating quantization graphs for {len(models_to_process)} models...")
    print(f"Database: {db.db_path}")
    print("-" * 60)
    
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    for model_name in models_to_process:
        if args.skip_existing and db.has_model_graph(model_name):
            print(f"⊘ Skipping {model_name} (already exists)")
            skip_count += 1
            continue
        
        if generate_graph_for_model(model_name, db, force=args.force, quantized=not args.vanilla):
            success_count += 1
        else:
            fail_count += 1
    
    # Print summary
    print("\n" + "-" * 60)
    print(f"✓ Completed: {success_count} graphs generated")
    if skip_count > 0:
        print(f"⊘ Skipped: {skip_count} graphs (already exist)")
    if fail_count > 0:
        print(f"✗ Failed: {fail_count} models")
    
    # Show storage info
    print("\n📦 Storage Summary:")
    graphs_df = db.get_all_model_graphs()
    if not graphs_df.empty:
        total_compressed = graphs_df['svg_size_compressed'].sum()
        total_original = graphs_df['svg_size_original'].sum()
        avg_reduction = 100 * (1 - total_compressed / total_original) if total_original > 0 else 0
        
        print(f"  Total graphs: {len(graphs_df)}")
        print(f"  Original size: {total_original / (1024*1024):.2f} MB")
        print(f"  Compressed size: {total_compressed / (1024*1024):.2f} MB")
        print(f"  Compression ratio: {avg_reduction:.1f}%")
        
        # Show per-model sizes
        print("\n  Per-model breakdown:")
        for _, row in graphs_df.iterrows():
            reduction = 100 * (1 - row['svg_size_compressed'] / row['svg_size_original']) if row['svg_size_original'] > 0 else 0
            print(f"    {row['model_name']:20s}: {row['svg_size_original']/1024:7.1f}KB → {row['svg_size_compressed']/1024:7.1f}KB ({reduction:5.1f}%)")

if __name__ == "__main__":
    main()
