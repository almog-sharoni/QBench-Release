import sys
import os

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.src.database.handler import RunDatabase

def test_handler():
    print("--- Testing RunDatabase (Refactored) ---")
    
    # Use a temporary database for testing
    test_db_path = "runspace/results/test_handler_runs.db"
    db = RunDatabase(db_path=test_db_path)
    
    # Clear any previous test data
    db.clear_database()
    
    # Log some dummy runs
    print("Logging dummy runs...")
    db.log_run("resnet18", "fp8_e4m3", "fp16", 75.32, 92.1, "SUCCESS")
    db.log_run("resnet50", "fp8_e5m2", "fp32", 78.41, 94.02, "SUCCESS")
    db.log_run("vit_b_16", "int8", "fp16", 0.0, 0.0, "FAILED")
    
    # Retrieve runs
    print("\nRetrieving runs as Pandas DataFrame:")
    df = db.get_runs()
    print(df)
    
    # Check shape
    assert len(df) == 3
    assert "model_name" in df.columns
    
    print("\nDatabase Summary:")
    print(db.get_summary())
    
    print("\n--- Handler Test Passed! ---")
    
    # Clean up test file
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

if __name__ == "__main__":
    test_handler()
