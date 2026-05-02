import os
import sys
PROJECT_ROOT = '/data/almog/Projects/QBench-Release'
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
from runspace.src.database.handler import RunDatabase
db = RunDatabase()
sim = db.get_latest_cache_simulation('vit_b_16')
if sim:
    names = [l['name'] for l in sim.get('layers', [])]
    print(f"Total layers in DB: {len(names)}")
    # Print the last 20 names, maybe they are at the end
    for n in names[-50:]:
        print(f"  {n}")
else:
    print("No simulation found")
