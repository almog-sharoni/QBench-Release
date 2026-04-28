import sqlite3
import os

db_path = "/data/almog/Projects/QBench-Release/runspace/database/runs.db"
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

cursor.execute("SELECT id, model_name, ref_acc1 FROM runs WHERE model_name = 'vit_b_16' AND ref_acc1 IS NOT NULL AND ref_acc1 != 0.0 LIMIT 5")
rows = cursor.fetchall()
for row in rows:
    print(f"ID: {row['id']} | Ref Acc: {row['ref_acc1']}")

conn.close()
