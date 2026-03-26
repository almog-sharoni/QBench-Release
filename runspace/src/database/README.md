# QBench Experiment Tracking System

This directory contains the core database and visualization components for tracking QBench experiment runs. The system uses a **SQLite** backend with a **Pandas**-compatible interface, making it robust enough for reliable logging yet flexible enough for deep data analysis.

## Key Components

| File | Description |
| :--- | :--- |
| `handler.py` | Contains the `RunDatabase` class — the primary interface for logging and retrieving experiment data. |
| `dashboard.py` | A **Streamlit**-based web application to visualize and compare results in real-time. |
| `test_handler.py` | A verification script to ensure the database engine is functioning correctly. |

---

## Getting Started

### 1. Integration: Logging a Run
To track your experiments, simply instantiate the `RunDatabase` class and call `log_run()` after each completion.

```python
from runspace.src.database.handler import RunDatabase

# Initialize (automatically creates runspace/results/runs.db if missing)
db = RunDatabase()

# Log an experiment result
db.log_run(
    model_name="resnet18",
    weight_dt="fp8_e4m3",
    activation_dt="fp16",
    acc1=75.32,
    acc5=92.1,
    status="SUCCESS"
)
```

### 2. Visualization: Running the Dashboard
The dashboard provides a clean web interface to filter, search, and visualize accuracy trends. I have provided a wrapper script to handle the Apptainer environment and port mapping.

```bash
# From the project root
./run_dashboard.sh
```

### 3. Verification
If you suspect issues with the logging system, run the test suite within your Apptainer container:

```bash
./run_apptainer.sh runspace/src/database/test_handler.py
```

---

## Data Schema

The database is stored in a single SQLite file at `runspace/results/runs.db`. It contains a `runs` table with the following structure:

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | INTEGER | Primary Key (autoincrement) |
| `model_name` | TEXT | Identifier for the model architecture |
| `weight_dt` | TEXT | Precision format used for weights |
| `activation_dt` | TEXT | Precision format used for activations |
| `acc1` | REAL | Top-1 Validation Accuracy (%) |
| `acc5` | REAL | Top-5 Validation Accuracy (%) |
| `run_date` | TEXT | Timestamp of the run completion |
| `status` | TEXT | Execution status (`SUCCESS`, `FAILED`, etc.) |

## Data Analysis with Pandas
Since the handler returns standard Pandas DataFrames, you can easily perform custom analysis in a notebook:

```python
from runspace.src.database.handler import RunDatabase
db = RunDatabase()
df = db.get_runs()

# Example: Get mean accuracy per weight type
summary = df.groupby('weight_dt')['acc1'].mean()
print(summary)
```
