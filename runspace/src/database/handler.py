import os
import re
import sqlite3
import pandas as pd
from datetime import datetime
import gzip
import json


def _parse_dt(dt_str):
    """Extract (bits, exp, mant) from strings like 'fp6_e1m4', 'fp32', etc."""
    if not dt_str or not isinstance(dt_str, str):
        return None, None, None
    s = dt_str.lower().strip()
    if s in ('fp32',): return 32, None, None
    if s in ('fp16',): return 16, None, None
    if s in ('bf16',): return 16, None, None
    bits = exp = mant = None
    parts = s.split('_')
    prefix = parts[0]
    for p in ('ufp', 'efp', 'fp'):
        if prefix.startswith(p):
            try: bits = int(prefix[len(p):])
            except ValueError: pass
            break
    if len(parts) > 1:
        em = parts[1]
        m = re.match(r'e(\d+)(?:m(\d+))?', em)
        if m:
            exp = int(m.group(1))
            mant = int(m.group(2)) if m.group(2) else None
    return bits, exp, mant


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

            # task_types lookup table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS task_types (
                    id   INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT    UNIQUE NOT NULL
                )
            ''')
            for name in ('classification', 'language_model', 'generic'):
                cursor.execute('INSERT OR IGNORE INTO task_types (name) VALUES (?)', (name,))

            # runs — flat metric columns, no JSON metrics fields
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS runs (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name      TEXT    NOT NULL,
                    experiment_type TEXT,
                    weight_dt       TEXT,
                    activation_dt   TEXT,
                    task_type_id    INTEGER REFERENCES task_types(id),
                    acc1            REAL,
                    acc5            REAL,
                    certainty       REAL,
                    ref_acc1        REAL,
                    ref_acc5        REAL,
                    ref_certainty   REAL,
                    mse             REAL,
                    l1              REAL,
                    w_bits          INTEGER,
                    w_exp           INTEGER,
                    w_mant          INTEGER,
                    a_bits          INTEGER,
                    a_exp           INTEGER,
                    a_mant          INTEGER,
                    status          TEXT,
                    run_date        TEXT,
                    quant_map_json  TEXT,
                    input_map_json  TEXT
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

    def _task_type_id(self, conn, task_type: str) -> int:
        c = conn.cursor()
        c.execute('INSERT OR IGNORE INTO task_types (name) VALUES (?)', (task_type,))
        c.execute('SELECT id FROM task_types WHERE name = ?', (task_type,))
        return c.fetchone()[0]

    def log_run(self, model_name, weight_dt, activation_dt, task_type,
                status="SUCCESS", experiment_type=None, run_date=None,
                # classification metrics
                acc1=None, acc5=None, certainty=None,
                ref_acc1=None, ref_acc5=None, ref_certainty=None,
                # error metrics
                mse=None, l1=None,
                # layer maps for dashboard
                quant_map_json=None, input_map_json=None):
        """
        Logs a new run to the database.
        quant_map_json : JSON string mapping layer -> weight format.
        input_map_json : JSON string mapping layer -> dominant input format
                         (from DynamicInputQuantizer layer_stats).
        """
        if not isinstance(task_type, str) or not task_type:
            raise ValueError("task_type must be a non-empty string")

        if run_date is None:
            run_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        w_bits, w_exp, w_mant = _parse_dt(weight_dt)
        a_bits, a_exp, a_mant = _parse_dt(activation_dt)

        with sqlite3.connect(self.db_path) as conn:
            task_type_id = self._task_type_id(conn, task_type)
            conn.execute('''
                INSERT INTO runs (
                    model_name, experiment_type, weight_dt, activation_dt, task_type_id,
                    acc1, acc5, certainty,
                    ref_acc1, ref_acc5, ref_certainty,
                    mse, l1,
                    w_bits, w_exp, w_mant,
                    a_bits, a_exp, a_mant,
                    status, run_date,
                    quant_map_json, input_map_json
                ) VALUES (
                    ?,?,?,?,?,
                    ?,?,?,
                    ?,?,?,
                    ?,?,
                    ?,?,?,
                    ?,?,?,
                    ?,?,
                    ?,?
                )
            ''', (
                model_name, experiment_type, weight_dt, activation_dt, task_type_id,
                acc1, acc5, certainty,
                ref_acc1, ref_acc5, ref_certainty,
                mse, l1,
                w_bits, w_exp, w_mant,
                a_bits, a_exp, a_mant,
                status, run_date,
                quant_map_json, input_map_json,
            ))
            conn.commit()
            print(f"Logged run for {model_name} to {self.db_path}")

    def get_runs(self, limit=None):
        """
        Retrieves runs as a Pandas DataFrame.

        Args:
            limit (int, optional): Max number of runs to return.

        Returns:
            pd.DataFrame: DataFrame containing run data, with task_type name column.
        """
        query = '''
            SELECT r.*, t.name AS task_type
            FROM runs r
            LEFT JOIN task_types t ON r.task_type_id = t.id
            ORDER BY r.id DESC
        '''
        if limit:
            query += f' LIMIT {limit}'
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn)

    def get_task_types(self):
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query('SELECT * FROM task_types ORDER BY name', conn)

    def clear_database(self):
        """Wipes all runs from the database (use with caution)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM runs")
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
