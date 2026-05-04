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
    @staticmethod
    def _default_db_path() -> str:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
        return os.path.join(project_root, "runspace/database/runs.db")

    def __init__(self, db_path=None):
        self.db_path = db_path if db_path is not None else self._default_db_path()
            
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
                    output_dt TEXT,
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
            try: cursor.execute("ALTER TABLE runs ADD COLUMN quant_map_json TEXT")
            except sqlite3.OperationalError: pass
            try: cursor.execute("ALTER TABLE runs ADD COLUMN input_map_json TEXT")
            except sqlite3.OperationalError: pass
            try: cursor.execute("ALTER TABLE runs ADD COLUMN output_map_json TEXT")
            except sqlite3.OperationalError: pass
            try: cursor.execute("ALTER TABLE runs ADD COLUMN output_dt TEXT")
            except sqlite3.OperationalError: pass
            try: cursor.execute("ALTER TABLE runs ADD COLUMN config_json TEXT")
            except sqlite3.OperationalError: pass
            
            # Create model_graphs table for storing quantization visualizations
            # Drop old table to clean up SVG space as user requested
            try:
                cursor.execute('PRAGMA user_version')
                version = cursor.fetchone()[0]
                if version < 1:
                    cursor.execute('DROP TABLE IF EXISTS model_graphs')
                    cursor.execute('PRAGMA user_version = 1')
            except sqlite3.OperationalError: pass

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_graphs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT UNIQUE NOT NULL,
                    graph_json_compressed BLOB,
                    graph_size_original INTEGER,
                    graph_size_compressed INTEGER,
                    metadata TEXT,
                    generated_date TEXT,
                    status TEXT
                )
            ''')
            
            try: cursor.execute("ALTER TABLE model_graphs ADD COLUMN metadata TEXT")
            except sqlite3.OperationalError: pass
            try: cursor.execute("ALTER TABLE model_graphs ADD COLUMN graph_size_original INTEGER")
            except sqlite3.OperationalError: pass
            try: cursor.execute("ALTER TABLE model_graphs ADD COLUMN graph_size_compressed INTEGER")
            except sqlite3.OperationalError: pass
            try: cursor.execute("ALTER TABLE model_graphs ADD COLUMN generated_date TEXT")
            except sqlite3.OperationalError: pass
            try: cursor.execute("ALTER TABLE model_graphs ADD COLUMN status TEXT")
            except sqlite3.OperationalError: pass

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fm_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    weight_dt TEXT,
                    activation_dt TEXT,
                    output_dt TEXT,
                    experiment_type TEXT,
                    run_date TEXT,
                    status TEXT,
                    fm_num_keypoints REAL,
                    fm_mean_score REAL,
                    fm_desc_norm REAL,
                    fm_repeatability REAL,
                    matching_precision REAL,
                    matching_score REAL,
                    mean_num_matches REAL,
                    pose_auc_5 REAL,
                    pose_auc_10 REAL,
                    pose_auc_20 REAL,
                    ref_matching_precision REAL,
                    ref_matching_score REAL,
                    ref_mean_num_matches REAL,
                    ref_pose_auc_5 REAL,
                    ref_pose_auc_10 REAL,
                    ref_pose_auc_20 REAL,
                    config_json TEXT
                )
            ''')

            try: cursor.execute("ALTER TABLE fm_runs ADD COLUMN output_dt TEXT")
            except sqlite3.OperationalError: pass
            try: cursor.execute("ALTER TABLE fm_runs ADD COLUMN output_map_json TEXT")
            except sqlite3.OperationalError: pass

            self._init_cache_sim_table(cursor)

            conn.commit()

    def log_run(self, model_name, weight_dt, activation_dt, acc1, acc5, status="SUCCESS",
                ref_acc1=None, ref_acc5=None, ref_certainty=None, experiment_type=None, run_date=None,
                mse=None, l1=None, certainty=None, quant_map_json=None, input_map_json=None,
                config_json=None, output_dt=None, output_map_json=None):
        """
        Logs a new run to the database.
        quant_map_json  : JSON string mapping layer -> weight format.
        input_map_json  : JSON string mapping layer -> dominant input format
                          (from DynamicInputQuantizer layer_stats).
        output_map_json : JSON string mapping layer -> output format / per-layer
                          override resolved at runtime.
        output_dt       : Global output quantization format string (e.g. fp8_e5m2);
                          'fp32' when output_quantization is disabled.
        config_json     : JSON string of the full run configuration dict.
        """
        if run_date is None:
            from datetime import datetime
            run_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO runs (
                    model_name, weight_dt, activation_dt, output_dt, acc1, acc5, ref_acc1, ref_acc5, ref_certainty,
                    experiment_type, run_date, status, mse, l1, certainty, quant_map_json, input_map_json,
                    output_map_json, config_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_name, weight_dt, activation_dt, output_dt, acc1, acc5, ref_acc1, ref_acc5, ref_certainty,
                experiment_type, run_date, status, mse, l1, certainty, quant_map_json, input_map_json,
                output_map_json, config_json
            ))
            conn.commit()
            print(f"Logged run for {model_name} to {self.db_path}")

    def run_exists(self, model_name, experiment_type=None, weight_dt=None, activation_dt=None, status="SUCCESS"):
        """Return True if at least one matching run exists."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            query = "SELECT COUNT(*) FROM runs WHERE model_name = ?"
            params = [model_name]

            if experiment_type is not None:
                query += " AND experiment_type = ?"
                params.append(experiment_type)
            if weight_dt is not None:
                query += " AND weight_dt = ?"
                params.append(weight_dt)
            if activation_dt is not None:
                query += " AND activation_dt = ?"
                params.append(activation_dt)
            if status is not None:
                query += " AND status = ?"
                params.append(status)

            cursor.execute(query, params)
            return cursor.fetchone()[0] > 0

    def get_latest_run(self, model_name, experiment_type=None, weight_dt=None, activation_dt=None, status="SUCCESS"):
        """Return latest matching run as dict, or None if not found."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            query = "SELECT * FROM runs WHERE model_name = ?"
            params = [model_name]

            if experiment_type is not None:
                query += " AND experiment_type = ?"
                params.append(experiment_type)
            if weight_dt is not None:
                query += " AND weight_dt = ?"
                params.append(weight_dt)
            if activation_dt is not None:
                query += " AND activation_dt = ?"
                params.append(activation_dt)
            if status is not None:
                query += " AND status = ?"
                params.append(status)

            query += " ORDER BY id DESC LIMIT 1"
            cursor.execute(query, params)
            row = cursor.fetchone()
            return dict(row) if row is not None else None

    def get_reference_metrics(self, model_name):
        """
        Returns latest fp32 reference tuple: (acc1, acc5, certainty), or None.
        Looks for successful fp32/fp32 runs, prioritizing explicit fp32_ref experiment type.
        If none found, searches for any run of this model that recorded non-zero reference metrics.
        """
        ref_row = self.get_latest_run(
            model_name=model_name,
            experiment_type="fp32_ref",
            weight_dt="fp32",
            activation_dt="fp32",
            status="SUCCESS",
        )
        if ref_row is None:
            ref_row = self.get_latest_run(
                model_name=model_name,
                weight_dt="fp32",
                activation_dt="fp32",
                status="SUCCESS",
            )
        
        if ref_row is None:
            # Fallback: Search for ANY run of this model that has ref_acc1 recorded
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT ref_acc1, ref_acc5, ref_certainty 
                    FROM runs 
                    WHERE model_name = ? AND ref_acc1 IS NOT NULL AND ref_acc1 != 0.0
                    ORDER BY id DESC LIMIT 1
                """, (model_name,))
                ref_row = cursor.fetchone()
                if ref_row:
                    ref_row = dict(ref_row)

        if ref_row is None:
            return None
            
        return (
            float(ref_row.get('ref_acc1', 0.0) or 0.0),
            float(ref_row.get('ref_acc5', 0.0) or 0.0),
            float(ref_row.get('ref_certainty', 0.0) or 0.0),
        )

    def get_best_run(self, model_name, experiment_type=None, weight_dt=None, activation_dt=None):
        """Return best-acc1 successful run as dict, or None if none found."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            query = "SELECT * FROM runs WHERE model_name = ? AND status = 'SUCCESS'"
            params = [model_name]

            if experiment_type is not None:
                query += " AND experiment_type = ?"
                params.append(experiment_type)
            if weight_dt is not None:
                query += " AND weight_dt = ?"
                params.append(weight_dt)
            if activation_dt is not None:
                query += " AND activation_dt = ?"
                params.append(activation_dt)

            query += " ORDER BY acc1 DESC, id DESC LIMIT 1"
            cursor.execute(query, params)
            row = cursor.fetchone()
            return dict(row) if row is not None else None

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

    def delete_runs_by_ids(self, run_ids):
        """Delete one or more runs by primary-key id and return deleted count."""
        if not run_ids:
            return 0

        normalized_ids = []
        for run_id in run_ids:
            if run_id is None:
                continue
            try:
                normalized_ids.append(int(run_id))
            except (TypeError, ValueError):
                continue

        normalized_ids = sorted(set(normalized_ids))
        if not normalized_ids:
            return 0

        placeholders = ",".join("?" for _ in normalized_ids)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"DELETE FROM runs WHERE id IN ({placeholders})",
                normalized_ids,
            )
            deleted_count = max(cursor.rowcount, 0)
            conn.commit()
            return deleted_count

    def update_experiment_type_by_ids(self, run_ids, experiment_type):
        """Update experiment_type for one or more run rows and return updated count."""
        if experiment_type is None:
            experiment_type = ""
        experiment_type = str(experiment_type).strip()

        normalized_ids = self._normalize_run_ids(run_ids)
        if not normalized_ids:
            return 0

        placeholders = ",".join("?" for _ in normalized_ids)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE runs SET experiment_type = ? WHERE id IN ({placeholders})",
                [experiment_type] + normalized_ids,
            )
            updated_count = max(cursor.rowcount, 0)
            conn.commit()
            return updated_count

    def create_database_from_run_ids(self, run_ids, destination_db_path):
        """
        Create a new SQLite run database containing the selected runs.

        The destination database is initialized with the current schema and then
        populated with matching columns from the source `runs` table. Existing
        destination files are rejected so accidental overwrites stay explicit.
        """
        normalized_ids = self._normalize_run_ids(run_ids)
        if not normalized_ids:
            return 0

        destination_db_path = os.path.abspath(destination_db_path)
        if os.path.exists(destination_db_path):
            raise FileExistsError(f"Destination database already exists: {destination_db_path}")

        os.makedirs(os.path.dirname(destination_db_path), exist_ok=True)
        destination_db = RunDatabase(db_path=destination_db_path)

        placeholders = ",".join("?" for _ in normalized_ids)
        with sqlite3.connect(self.db_path) as src_conn, sqlite3.connect(destination_db.db_path) as dst_conn:
            src_conn.row_factory = sqlite3.Row
            src_cursor = src_conn.cursor()
            dst_cursor = dst_conn.cursor()

            src_cursor.execute(f"SELECT * FROM runs WHERE id IN ({placeholders})", normalized_ids)
            rows = [dict(row) for row in src_cursor.fetchall()]
            if not rows:
                return 0

            dst_cursor.execute("PRAGMA table_info(runs)")
            destination_cols = [row[1] for row in dst_cursor.fetchall()]
            insert_cols = [col for col in destination_cols if col in rows[0]]
            if not insert_cols:
                return 0

            col_sql = ", ".join(insert_cols)
            value_sql = ", ".join("?" for _ in insert_cols)
            dst_cursor.executemany(
                f"INSERT INTO runs ({col_sql}) VALUES ({value_sql})",
                [[row.get(col) for col in insert_cols] for row in rows],
            )
            dst_conn.commit()
            return len(rows)

    @staticmethod
    def _normalize_run_ids(run_ids):
        normalized_ids = []
        for run_id in run_ids or []:
            if run_id is None:
                continue
            try:
                normalized_ids.append(int(run_id))
            except (TypeError, ValueError):
                continue
        return sorted(set(normalized_ids))

    def get_summary(self):
        """Returns a summary of runs grouped by model and status."""
        df = self.get_runs()
        if df.empty:
            return "No runs found."
        return df.groupby(['model_name', 'status']).size().unstack(fill_value=0)

    # ============= FEATURE MATCHING METHODS =============

    def log_fm_run(self, model_name, weight_dt, activation_dt, experiment_type=None,
                   status="SUCCESS", run_date=None,
                   fm_num_keypoints=None, fm_mean_score=None, fm_desc_norm=None,
                   fm_repeatability=None, matching_precision=None, matching_score=None,
                   mean_num_matches=None, pose_auc_5=None, pose_auc_10=None,
                   pose_auc_20=None, ref_matching_precision=None, ref_matching_score=None,
                   ref_mean_num_matches=None, ref_pose_auc_5=None, ref_pose_auc_10=None,
                   ref_pose_auc_20=None, config_json=None,
                   output_dt=None, output_map_json=None):
        """Logs a feature-matching run to the fm_runs table."""
        if run_date is None:
            run_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO fm_runs (
                    model_name, weight_dt, activation_dt, output_dt, experiment_type, run_date, status,
                    fm_num_keypoints, fm_mean_score, fm_desc_norm, fm_repeatability,
                    matching_precision, matching_score, mean_num_matches,
                    pose_auc_5, pose_auc_10, pose_auc_20,
                    ref_matching_precision, ref_matching_score, ref_mean_num_matches,
                    ref_pose_auc_5, ref_pose_auc_10, ref_pose_auc_20,
                    output_map_json, config_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_name, weight_dt, activation_dt, output_dt, experiment_type, run_date, status,
                fm_num_keypoints, fm_mean_score, fm_desc_norm, fm_repeatability,
                matching_precision, matching_score, mean_num_matches,
                pose_auc_5, pose_auc_10, pose_auc_20,
                ref_matching_precision, ref_matching_score, ref_mean_num_matches,
                ref_pose_auc_5, ref_pose_auc_10, ref_pose_auc_20,
                output_map_json, config_json,
            ))
            conn.commit()
            print(f"Logged FM run for {model_name} to {self.db_path}")

    def fm_run_exists(self, model_name, experiment_type=None, weight_dt=None,
                      activation_dt=None, status="SUCCESS"):
        """Return True if a matching fm_runs row exists."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            query = "SELECT COUNT(*) FROM fm_runs WHERE model_name = ?"
            params = [model_name]
            if experiment_type is not None:
                query += " AND experiment_type = ?"
                params.append(experiment_type)
            if weight_dt is not None:
                query += " AND weight_dt = ?"
                params.append(weight_dt)
            if activation_dt is not None:
                query += " AND activation_dt = ?"
                params.append(activation_dt)
            if status is not None:
                query += " AND status = ?"
                params.append(status)
            cursor.execute(query, params)
            return cursor.fetchone()[0] > 0

    def get_fm_runs(self, limit=None):
        """Returns fm_runs as a Pandas DataFrame."""
        query = "SELECT * FROM fm_runs ORDER BY id DESC"
        if limit:
            query += f" LIMIT {limit}"
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn)

    # ============= MODEL GRAPH METHODS =============
    
    def store_model_graph(self, model_name, graph_json_content, metadata=None):
        """
        Stores a quantization graph for a model.
        Compresses JSON with gzip for efficient storage.
        
        Args:
            model_name (str): Name of the model
            graph_json_content (str): JSON content as string
            metadata (dict, optional): Additional metadata (layer_count, quant_count, etc.)
        """
        if metadata is None:
            metadata = {}
        
        # Compress JSON using gzip
        json_bytes = graph_json_content.encode('utf-8')
        compressed = gzip.compress(json_bytes, compresslevel=9)  # Max compression
        
        # Store metadata as JSON
        metadata_json = json.dumps(metadata)
        
        original_size = len(json_bytes)
        compressed_size = len(compressed)
        generated_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        compression_ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Use INSERT OR REPLACE for idempotency
            cursor.execute('''
                INSERT OR REPLACE INTO model_graphs 
                (model_name, graph_json_compressed, graph_size_original, graph_size_compressed, 
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
    
    def get_model_graph_json(self, model_name):
        """
        Retrieves and decompresses graph JSON for a model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            tuple: (graph_json_content, metadata) or (None, None) if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT graph_json_compressed, metadata, generated_date, status FROM model_graphs 
                WHERE model_name = ?
            ''', (model_name,))
            result = cursor.fetchone()
            
            if result:
                compressed, metadata_json, generated_date, status = result
                graph_json_content = gzip.decompress(compressed).decode('utf-8')
                metadata = json.loads(metadata_json) if metadata_json else {}
                metadata['generated_date'] = generated_date
                metadata['status'] = status
                return graph_json_content, metadata
            return None, None
    
    def get_model_graph_metadata(self, model_name):
        """
        Retrieves graph metadata without decompressing JSON.
        Useful for listing graph info without loading full JSON.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            dict: Metadata dict with graph_size_original, graph_size_compressed, generated_date, etc.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT graph_size_original, graph_size_compressed, metadata, 
                       generated_date, status FROM model_graphs 
                WHERE model_name = ?
            ''', (model_name,))
            result = cursor.fetchone()
            
            if result:
                size_orig, size_comp, metadata_json, gen_date, status = result
                metadata = json.loads(metadata_json) if metadata_json else {}
                return {
                    'model_name': model_name,
                    'graph_size_original': size_orig,
                    'graph_size_compressed': size_comp,
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
                SELECT model_name, graph_size_original, graph_size_compressed, 
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

    # ============= CACHE SIMULATION METHODS =============

    def _init_cache_sim_table(self, cursor):
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache_simulations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                cache_size_M REAL,
                num_banks INTEGER,
                bank_size INTEGER,
                metadata_bits INTEGER,
                batch_size INTEGER,
                total_layers INTEGER,
                off_chip_count INTEGER,
                flagged_count INTEGER,
                timestamp TEXT,
                layers_json TEXT,
                off_chip_layers_json TEXT,
                rules_json TEXT
            )
        ''')
        try:
            cursor.execute("ALTER TABLE cache_simulations ADD COLUMN rules_json TEXT")
        except sqlite3.OperationalError:
            pass

    def store_cache_simulation(self, sim_data: dict):
        """
        Store a cache simulation result.

        Args:
            sim_data: The full dict produced by run_simulation (keys: metadata, summary, layers, off_chip_layers).
        """
        meta    = sim_data.get('metadata', {})
        summary = sim_data.get('summary', {})

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            self._init_cache_sim_table(cursor)
            cursor.execute('''
                INSERT INTO cache_simulations (
                    model_name, cache_size_M, num_banks, bank_size, metadata_bits, batch_size,
                    total_layers, off_chip_count, flagged_count, timestamp,
                    layers_json, off_chip_layers_json, rules_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                meta.get('model'),
                meta.get('cache_size_M'),
                meta.get('num_banks'),
                meta.get('bank_size'),
                meta.get('metadata_bits'),
                meta.get('batch_size'),
                summary.get('total_layers'),
                summary.get('quantize_count'),
                summary.get('flagged_count'),
                meta.get('timestamp'),
                json.dumps(sim_data.get('layers', [])),
                json.dumps(sim_data.get('off_chip_layers', [])),
                json.dumps(sim_data.get('rules', [])),
            ))
            conn.commit()
            print(f"[CacheSim] Stored simulation for {meta.get('model')} to DB.")

    def get_cache_simulations(self) -> pd.DataFrame:
        """Return all cache simulation runs as a DataFrame (latest first)."""
        with sqlite3.connect(self.db_path) as conn:
            try:
                df = pd.read_sql_query(
                    'SELECT * FROM cache_simulations ORDER BY id DESC', conn
                )
            except Exception:
                df = pd.DataFrame()
        return df

    def get_latest_cache_simulation(self, model_name: str) -> dict | None:
        """Return the most recent simulation result for a model as a dict, or None."""
        with sqlite3.connect(self.db_path) as conn:
            try:
                self._init_cache_sim_table(conn.cursor())
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT * FROM cache_simulations WHERE model_name = ? ORDER BY id DESC LIMIT 1',
                    (model_name,)
                )
                row = cursor.fetchone()
            except Exception:
                row = None
        if row is None:
            return None
        d = dict(row)
        d['layers']          = json.loads(d.get('layers_json') or '[]')
        d['off_chip_layers'] = json.loads(d.get('off_chip_layers_json') or '[]')
        return d
