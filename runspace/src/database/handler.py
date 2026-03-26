import os
import sqlite3
import pandas as pd
from datetime import datetime

class RunDatabase:
    """
    A simple database handler for tracking experiment runs.
    Uses SQLite backend with Pandas-compatible methods.
    """
    def __init__(self, db_path=None):
        if db_path is None:
            # Default path relative to project root
            # src/database/handler.py -> src/database -> src -> runspace -> root
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
            self.db_path = os.path.join(project_root, "runspace/database/runs.db")
        else:
            self.db_path = db_path
            
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the database schema if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    weight_dt TEXT,
                    activation_dt TEXT,
                    acc1 REAL,
                    acc5 REAL,
                    ref_acc1 REAL,
                    ref_acc5 REAL,
                    ref_certainty REAL,
                    experiment_type TEXT,
                    run_date TEXT,
                    status TEXT,
                    mse REAL,
                    l1 REAL,
                    certainty REAL
                )
            ''')
            
            # Migration: Check if columns exist and add if not (for existing databases)
            try:
                cursor.execute("ALTER TABLE runs ADD COLUMN ref_acc1 REAL")
                cursor.execute("ALTER TABLE runs ADD COLUMN ref_acc5 REAL")
            except sqlite3.OperationalError: pass # Already exists
            
            try:
                cursor.execute("ALTER TABLE runs ADD COLUMN experiment_type TEXT")
            except sqlite3.OperationalError: pass # Already exists

            try: cursor.execute("ALTER TABLE runs ADD COLUMN ref_acc1 REAL")
            except sqlite3.OperationalError: pass
            try: cursor.execute("ALTER TABLE runs ADD COLUMN ref_acc5 REAL")
            except sqlite3.OperationalError: pass
            try: cursor.execute("ALTER TABLE runs ADD COLUMN ref_certainty REAL")
            except sqlite3.OperationalError: pass
            try: cursor.execute("ALTER TABLE runs ADD COLUMN run_date TEXT")
            except sqlite3.OperationalError: pass
            try: cursor.execute("ALTER TABLE runs ADD COLUMN status TEXT")
            except sqlite3.OperationalError: pass
            try: cursor.execute("ALTER TABLE runs ADD COLUMN mse REAL")
            except sqlite3.OperationalError: pass
            try: cursor.execute("ALTER TABLE runs ADD COLUMN l1 REAL")
            except sqlite3.OperationalError: pass
            try: cursor.execute("ALTER TABLE runs ADD COLUMN certainty REAL")
            except sqlite3.OperationalError: pass
                
            conn.commit()

    def log_run(self, model_name, weight_dt, activation_dt, acc1, acc5, status="SUCCESS", 
                ref_acc1=None, ref_acc5=None, ref_certainty=None, experiment_type=None, run_date=None,
                mse=None, l1=None, certainty=None):
        """
        Logs a new run to the database.
        """
        if run_date is None:
            from datetime import datetime
            run_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO runs (
                    model_name, weight_dt, activation_dt, acc1, acc5, ref_acc1, ref_acc5, ref_certainty,
                    experiment_type, run_date, status, mse, l1, certainty
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_name, weight_dt, activation_dt, acc1, acc5, ref_acc1, ref_acc5, ref_certainty,
                experiment_type, run_date, status, mse, l1, certainty
            ))
            conn.commit()
            print(f"Logged run for {model_name} to {self.db_path}")

    def get_runs(self, limit=None):
        """
        Retrieves runs as a Pandas DataFrame.
        
        Args:
            limit (int, optional): Max number of runs to return.
            
        Returns:
            pd.DataFrame: DataFrame containing run data.
        """
        query = "SELECT * FROM runs ORDER BY id DESC"
        if limit:
            query += f" LIMIT {limit}"
            
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn)
            return df

    def clear_database(self):
        """Wipes all runs from the database (use with caution)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM runs")
            conn.commit()
            print("Database cleared.")

    def get_summary(self):
        """Returns a summary of runs grouped by model and status."""
        df = self.get_runs()
        if df.empty:
            return "No runs found."
        return df.groupby(['model_name', 'status']).size().unstack(fill_value=0)
