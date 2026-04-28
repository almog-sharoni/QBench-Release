import sqlite3
import os

db_path = "/data/almog/Projects/QBench-Release/runspace/database/runs.db"
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

cursor.execute("SELECT id, model_name, weight_dt, activation_dt, acc1, experiment_type FROM runs WHERE weight_dt = 'fp32' OR activation_dt = 'fp32' LIMIT 20")
rows = cursor.fetchall()
for row in rows:
    print(f"ID: {row['id']} | Model: {row['model_name']} | W: {row['weight_dt']} | A: {row['activation_dt']} | Acc: {row['acc1']} | Exp: {row['experiment_type']}")

conn.close()
