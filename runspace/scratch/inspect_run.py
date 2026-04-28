import sqlite3
import os

db_path = "/data/almog/Projects/QBench-Release/runspace/database/runs.db"
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

cursor.execute("SELECT * FROM runs WHERE id = 353")
row = cursor.fetchone()
for k in row.keys():
    print(f"{k}: {row[k]}")

conn.close()
