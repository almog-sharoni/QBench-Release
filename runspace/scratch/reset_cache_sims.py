import sqlite3
import os
import subprocess

DB_PATH = "runspace/database/runs.db"
MODELS = ["resnet18", "resnet152", "mobilevit_s", "vit_b_16", "resnext101_64x4d", "mobilenet_v3_large"]

def main():
    # 1. Clear database (just in case)
    print("Ensuring cache_simulations table is cleared...")
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM cache_simulations")
            conn.commit()
            print("Database cleared.")
        except sqlite3.OperationalError:
            print("Table cache_simulations does not exist yet.")
        conn.close()

    # 2. Re-run simulations using apptainer
    for model in MODELS:
        print(f"\n>>> Running cache simulation for: {model}")
        # Note: apptainer.sh prepends 'python' to the command
        cmd = [
            "./apptainer.sh", "runspace/experiments/asic_cache_simulation/simulate_cache.py",
            "--model_name", model
        ]
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully re-simulated {model}")
        except subprocess.CalledProcessError as e:
            print(f"Error simulating {model}: {e}")

if __name__ == "__main__":
    main()
