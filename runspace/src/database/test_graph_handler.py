#!/usr/bin/env python3
"""
Test script for model graph storage and visualization features.
Validates that the database, compression, and retrieval work correctly.
"""

import os
import sys
import tempfile
import gzip

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.src.database.handler import RunDatabase

def test_graph_storage():
    """Test storing and retrieving compressed graphs."""
    print("=" * 60)
    print("Testing Model Graph Storage")
    print("=" * 60)
    
    # Use temporary database for testing
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        test_db_path = f.name
    
    try:
        db = RunDatabase(db_path=test_db_path)
        
        # Create a simple SVG for testing
        test_svg = """<?xml version="1.0" encoding="UTF-8"?>
        <svg xmlns="http://www.w3.org/2000/svg" width="400" height="300">
          <rect x="10" y="10" width="100" height="50" fill="#90EE90" stroke="black"/>
          <text x="20" y="40" font-family="Arial">Quantized Layer</text>
          
          <rect x="150" y="10" width="100" height="50" fill="#FFD700" stroke="black"/>
          <text x="160" y="40" font-family="Arial">Supported</text>
          
          <rect x="290" y="10" width="100" height="50" fill="#FFB6C1" stroke="black"/>
          <text x="300" y="40" font-family="Arial">Unsupported</text>
          
          <rect x="75" y="100" width="250" height="100" fill="#D3D3D3" stroke="black"/>
          <text x="85" y="140" font-family="Arial">Test Model Graph</text>
          <text x="85" y="160" font-family="Arial">Multiple layers shown</text>
          <text x="85" y="180" font-family="Arial">with compression demo</text>
        </svg>"""
        
        print("\n1️⃣  Testing graph storage...")
        original_size = len(test_svg.encode('utf-8'))
        
        metadata = {
            'num_nodes': 42,
            'num_quantized_layers': 12,
            'test': True,
            'framework': 'pytorch'
        }
        
        db.store_model_graph('test_model', test_svg, metadata)
        print(f"   ✓ Stored test graph ({original_size} bytes)")
        
        print("\n2️⃣  Testing graph retrieval...")
        retrieved_svg, retrieved_meta = db.get_model_graph_svg('test_model')
        
        if retrieved_svg == test_svg:
            print(f"   ✓ Retrieved SVG matches original")
        else:
            print(f"   ✗ Retrieved SVG doesn't match!")
            return False
        
        if retrieved_meta == metadata:
            print(f"   ✓ Metadata matches original")
        else:
            print(f"   ✗ Metadata doesn't match! Got: {retrieved_meta}")
            return False
        
        print("\n3️⃣  Testing metadata-only retrieval...")
        meta_only = db.get_model_graph_metadata('test_model')
        
        if meta_only:
            print(f"   ✓ Got metadata without SVG")
            print(f"     - Original size: {meta_only['svg_size_original']} bytes")
            print(f"     - Compressed size: {meta_only['svg_size_compressed']} bytes")
            
            compression_ratio = (1 - meta_only['svg_size_compressed'] / meta_only['svg_size_original']) * 100
            print(f"     - Compression ratio: {compression_ratio:.1f}%")
            
            if compression_ratio < 0:
                print(f"   ⚠️  Small content doesn't compress well (expected for small SVGs)")
        else:
            print(f"   ✗ Failed to get metadata")
            return False
        
        print("\n4️⃣  Testing model existence check...")
        has_test = db.has_model_graph('test_model')
        has_nonexist = db.has_model_graph('nonexistent_model')
        
        if has_test and not has_nonexist:
            print(f"   ✓ Existence checks work correctly")
        else:
            print(f"   ✗ Existence check failed!")
            return False
        
        print("\n5️⃣  Testing list all graphs...")
        all_graphs = db.get_all_model_graphs()
        
        if len(all_graphs) == 1 and all_graphs.iloc[0]['model_name'] == 'test_model':
            print(f"   ✓ Listed all graphs successfully")
            print(f"     Found {len(all_graphs)} graph(s)")
        else:
            print(f"   ✗ List operation failed! Got: {all_graphs}")
            return False
        
        print("\n6️⃣  Testing update/replace...")
        new_svg = test_svg + "<!-- Updated -->"
        db.store_model_graph('test_model', new_svg, {'updated': True})
        
        retrieved_svg2, retrieved_meta2 = db.get_model_graph_svg('test_model')
        if retrieved_svg2 == new_svg and retrieved_meta2.get('updated'):
            print(f"   ✓ Update operation successful")
        else:
            print(f"   ✗ Update operation failed!")
            return False
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
            print(f"\nCleaned up test database: {test_db_path}")

if __name__ == "__main__":
    success = test_graph_storage()
    sys.exit(0 if success else 1)
