import sqlite3
import json
import os

db_path = "/data/almog/Projects/QBench-Release/runspace/database/runs.db"
if not os.path.exists(db_path):
    print(f"DB not found at {db_path}")
    exit(1)

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

cursor.execute("SELECT id, model_name, experiment_type, weight_dt, activation_dt, quant_map_json, input_map_json FROM runs ORDER BY id DESC LIMIT 5")
rows = cursor.fetchall()

for row in rows:
    print(f"ID: {row['id']}")
    print(f"Model: {row['model_name']}")
    print(f"Exp: {row['experiment_type']}")
    print(f"Activation DT: {row['activation_dt']}")
    
    input_map = row['input_map_json']
    if input_map:
        # Search for 'ufp' in the json
        if 'ufp' in input_map:
            print("FOUND 'ufp' in input_map_json")
        else:
            print("No 'ufp' in input_map_json")
    else:
        print("input_map_json is NULL")
    
    quant_map = row['quant_map_json']
    if quant_map:
        if 'ufp' in quant_map:
            print("FOUND 'ufp' in quant_map_json")
        else:
            print("No 'ufp' in quant_map_json")
    else:
        print("quant_map_json is NULL")
    print("-" * 20)

conn.close()
