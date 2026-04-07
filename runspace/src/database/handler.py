import os
import sqlite3
import pandas as pd
from datetime import datetime
import gzip
import base64
import json

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
                    task_type TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    ref_metrics_json TEXT,
                    experiment_type TEXT,
                    run_date TEXT,
                    status TEXT,
                    mse REAL,
                    l1 REAL,
                    quant_map_json TEXT,
                    input_map_json TEXT
                )
            ''')
            
            # Create model_graphs table for storing quantization visualizations
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_graphs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT UNIQUE NOT NULL,
                    svg_compressed BLOB,
                    svg_size_original INTEGER,
                    svg_size_compressed INTEGER,
                    metadata TEXT,
                    generated_date TEXT,
                    status TEXT
                )
            ''')
            
            try: cursor.execute("ALTER TABLE model_graphs ADD COLUMN metadata TEXT")
            except sqlite3.OperationalError: pass
            try: cursor.execute("ALTER TABLE model_graphs ADD COLUMN svg_size_original INTEGER")
            except sqlite3.OperationalError: pass
            try: cursor.execute("ALTER TABLE model_graphs ADD COLUMN svg_size_compressed INTEGER")
            except sqlite3.OperationalError: pass
            try: cursor.execute("ALTER TABLE model_graphs ADD COLUMN generated_date TEXT")
            except sqlite3.OperationalError: pass
            try: cursor.execute("ALTER TABLE model_graphs ADD COLUMN status TEXT")
            except sqlite3.OperationalError: pass
                
            conn.commit()

    def log_run(self, model_name, weight_dt, activation_dt, task_type, metrics, status="SUCCESS",
                ref_metrics=None, experiment_type=None, run_date=None,
                mse=None, l1=None, quant_map_json=None, input_map_json=None):
        """
        Logs a new run to the database.

        task_type     : Required. Task identifier, e.g. "classification" or "language_model".
        metrics       : Required. Dict of metric name -> float for the quantized run,
                        e.g. {"acc1": 76.3, "acc5": 93.0} or {"ppl": 12.4}.
        ref_metrics   : Optional. Same structure for the reference (fp32) run.
        quant_map_json: JSON string mapping layer -> weight format.
        input_map_json: JSON string mapping layer -> dominant input format
                        (from DynamicInputQuantizer layer_stats).
        """
        if not isinstance(task_type, str) or not task_type:
            raise ValueError("task_type must be a non-empty string")
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError("metrics must be a non-empty dict")

        if run_date is None:
            run_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        metrics_json = json.dumps(metrics)
        ref_metrics_json = json.dumps(ref_metrics) if ref_metrics is not None else None

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO runs (
                    model_name, weight_dt, activation_dt, task_type, metrics_json, ref_metrics_json,
                    experiment_type, run_date, status, mse, l1, quant_map_json, input_map_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_name, weight_dt, activation_dt, task_type, metrics_json, ref_metrics_json,
                experiment_type, run_date, status, mse, l1, quant_map_json, input_map_json
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
        """Returns a summary of runs grouped by model, task type, and status."""
        df = self.get_runs()
        if df.empty:
            return "No runs found."
        return df.groupby(['model_name', 'task_type', 'status']).size().unstack(fill_value=0)

    # ============= MODEL GRAPH METHODS =============
    
    def store_model_graph(self, model_name, svg_content, metadata=None):
        """
        Stores a quantization graph for a model.
        Compresses SVG with gzip for efficient storage.
        
        Args:
            model_name (str): Name of the model
            svg_content (str): SVG content as string
            metadata (dict, optional): Additional metadata (layer_count, quant_count, etc.)
        """
        if metadata is None:
            metadata = {}
        
        # Compress SVG using gzip
        svg_bytes = svg_content.encode('utf-8')
        compressed = gzip.compress(svg_bytes, compresslevel=9)  # Max compression
        
        # Store metadata as JSON
        metadata_json = json.dumps(metadata)
        
        original_size = len(svg_bytes)
        compressed_size = len(compressed)
        generated_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        compression_ratio = (1 - compressed_size / original_size) * 100
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Use INSERT OR REPLACE for idempotency
            cursor.execute('''
                INSERT OR REPLACE INTO model_graphs 
                (model_name, svg_compressed, svg_size_original, svg_size_compressed, 
                 metadata, generated_date, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_name, compressed, original_size, compressed_size,
                metadata_json, generated_date, "SUCCESS"
            ))
            conn.commit()
            print(f"Stored graph for {model_name}: "
                  f"{original_size/1024:.1f}KB → {compressed_size/1024:.1f}KB "
                  f"({compression_ratio:.1f}% reduction)")
    
    def get_model_graph_svg(self, model_name):
        """
        Retrieves and decompresses SVG for a model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            tuple: (svg_content, metadata) or (None, None) if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT svg_compressed, metadata FROM model_graphs 
                WHERE model_name = ?
            ''', (model_name,))
            result = cursor.fetchone()
            
            if result:
                compressed, metadata_json = result
                svg_content = gzip.decompress(compressed).decode('utf-8')
                metadata = json.loads(metadata_json) if metadata_json else {}
                return svg_content, metadata
            return None, None
    
    def get_model_graph_metadata(self, model_name):
        """
        Retrieves graph metadata without decompressing SVG.
        Useful for listing graph info without loading full SVG.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            dict: Metadata dict with svg_size_original, svg_size_compressed, generated_date, etc.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT svg_size_original, svg_size_compressed, metadata, 
                       generated_date, status FROM model_graphs 
                WHERE model_name = ?
            ''', (model_name,))
            result = cursor.fetchone()
            
            if result:
                size_orig, size_comp, metadata_json, gen_date, status = result
                metadata = json.loads(metadata_json) if metadata_json else {}
                return {
                    'model_name': model_name,
                    'svg_size_original': size_orig,
                    'svg_size_compressed': size_comp,
                    'generated_date': gen_date,
                    'status': status,
                    **metadata
                }
            return None
    
    def get_all_model_graphs(self):
        """
        Retrieves metadata for all stored model graphs.
        
        Returns:
            pd.DataFrame: DataFrame with graph metadata
        """
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query('''
                SELECT model_name, svg_size_original, svg_size_compressed, 
                       metadata, generated_date, status FROM model_graphs 
                ORDER BY generated_date DESC
            ''', conn)
            
            # Parse metadata JSON for each row
            if not df.empty:
                df['metadata_dict'] = df['metadata'].apply(lambda x: json.loads(x) if x else {})
            
            return df
    
    def has_model_graph(self, model_name):
        """Check if a graph exists for a model."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM model_graphs WHERE model_name = ?', (model_name,))
            return cursor.fetchone()[0] > 0
