import sqlite3
import json
import os

db_path = "/data/almog/Projects/QBench-Release/runspace/database/runs.db"
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

cursor.execute("SELECT id, model_name, experiment_type, quant_map_json FROM runs ORDER BY id DESC LIMIT 1")
row = cursor.fetchone()

print(f"ID: {row['id']}")
print(f"Model: {row['model_name']}")
print(f"Exp: {row['experiment_type']}")
quant_map = row['quant_map_json']

if quant_map:
    # Check if 'ufp' exists anywhere in the JSON
    if 'ufp' in quant_map:
        print("FOUND 'ufp' in quant_map_json!")
        # Print a snippet of where it was found
        idx = quant_map.find('ufp')
        print(f"Snippet: ...{quant_map[max(0, idx-20):idx+30]}...")
    else:
        print("No 'ufp' found in quant_map_json.")
        # print(quant_map[:500])
else:
    print("quant_map_json is NULL")

conn.close()
