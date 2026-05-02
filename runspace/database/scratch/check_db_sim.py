import os
import sys
PROJECT_ROOT = '/data/almog/Projects/QBench-Release'
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
from runspace.src.database.handler import RunDatabase
db = RunDatabase()
sim = db.get_latest_cache_simulation('vit_b_16')
if sim:
    print(f"Found simulation with {len(sim.get('layers', []))} layers.")
    for l in sim.get('layers', [])[:10]:
        print(f"  {l['name']}: {l['stay_on_chip']}")
else:
    print("No simulation found for vit_b_16")
