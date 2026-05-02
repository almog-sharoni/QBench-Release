import os
import sys
PROJECT_ROOT = '/data/almog/Projects/QBench-Release'
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
from runspace.src.database.handler import RunDatabase
db = RunDatabase()
sim = db.get_latest_cache_simulation('vit_b_16')
if sim:
    names = [l['name'] for l in sim.get('layers', [])]
    print(f"Total layers: {len(names)}")
    found = 0
    for n in names:
        if 'matmul' in n.lower() or 'div' in n.lower():
            print(f"  {n}")
            found += 1
    print(f"Found {found} matches")
else:
    print("No simulation found")
