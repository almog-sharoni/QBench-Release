import sqlite3
import os

db_path = "/data/almog/Projects/QBench-Release/runspace/database/runs.db"
if not os.path.exists(db_path):
    print(f"DB not found at {db_path}")
    exit(1)

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# 1. Map models to best available reference metrics
# We search for ANY run that recorded ref_acc1 (from a compare pass)
cursor.execute("""
    SELECT model_name, ref_acc1, ref_acc5, ref_certainty 
    FROM runs 
    WHERE ref_acc1 IS NOT NULL AND ref_acc1 != 0.0
    ORDER BY id DESC
""")
ref_map = {}
for row in cursor.fetchall():
    if row['model_name'] not in ref_map:
        ref_map[row['model_name']] = (row['ref_acc1'], row['ref_acc5'], row['ref_certainty'])

# Also check for explicit fp32/fp32 runs that might have acc1 but not ref_acc1 (they are the ref!)
cursor.execute("""
    SELECT model_name, acc1, acc5, certainty 
    FROM runs 
    WHERE weight_dt = 'fp32' AND activation_dt = 'fp32' AND status = 'SUCCESS'
    ORDER BY id DESC
""")
for row in cursor.fetchall():
    if row['model_name'] not in ref_map:
        ref_map[row['model_name']] = (row['acc1'], row['acc5'], row['certainty'])

print(f"Found reference metrics for models: {list(ref_map.keys())}")

# 2. Find runs with missing ref metrics (0.0 or NULL)
cursor.execute("""
    SELECT id, model_name, acc1, ref_acc1 
    FROM runs 
    WHERE (ref_acc1 IS NULL OR ref_acc1 = 0.0) 
      AND (weight_dt != 'fp32' OR activation_dt != 'fp32')
""")
runs_to_fix = cursor.fetchall()
print(f"Found {len(runs_to_fix)} runs to fix.")

fixed_count = 0
for run in runs_to_fix:
    model = run['model_name']
    if model in ref_map:
        ref_acc1, ref_acc5, ref_cert = ref_map[model]
        cursor.execute("""
            UPDATE runs 
            SET ref_acc1 = ?, ref_acc5 = ?, ref_certainty = ? 
            WHERE id = ?
        """, (ref_acc1, ref_acc5, ref_cert, run['id']))
        fixed_count += 1

conn.commit()
conn.close()
print(f"Successfully updated {fixed_count} runs with reference metrics.")
