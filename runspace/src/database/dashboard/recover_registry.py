
import sys
import os

file_path = "/data/almog/Projects/QBench-Release/runspace/src/database/dashboard/run_models_tab.py"

with open(file_path, "r") as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    # Add json import
    if "import glob" in line:
        new_lines.append(line)
        new_lines.append("    import json\n")
        continue
    
    # Add RUNNER_REGISTRY_PATH and Registry Functions
    if "RUNNER_OUTPUT_DIR =" in line:
        new_lines.append(line)
        new_lines.append("    RUNNER_REGISTRY_PATH = os.path.join(RUNNER_OUTPUT_DIR, 'registry.json')\n")
        
        registry_code = """
    def _dashboard_runner_pid_running(pid):
        if not pid: return False
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError): return False
        except Exception: return False

    def _dashboard_runner_load_registry():
        if not os.path.exists(RUNNER_REGISTRY_PATH): return []
        try:
            with open(RUNNER_REGISTRY_PATH, "r") as f:
                return json.load(f)
        except Exception: return []

    def _dashboard_runner_save_registry(registry):
        os.makedirs(os.path.dirname(RUNNER_REGISTRY_PATH), exist_ok=True)
        tmp_path = f"{RUNNER_REGISTRY_PATH}.tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(registry, f, indent=4)
            os.replace(tmp_path, RUNNER_REGISTRY_PATH)
        except Exception as e:
            st.error(f"Failed to save registry: {e}")

    def _dashboard_runner_refresh_registry():
        registry = _dashboard_runner_load_registry()
        changed = False
        for run in registry:
            if run.get("status") == "running":
                if not _dashboard_runner_pid_running(run.get("pid")):
                    run["status"] = "finished"
                    changed = True
        if changed: _dashboard_runner_save_registry(registry)
        return registry

    def _dashboard_runner_current_process():
        registry = _dashboard_runner_refresh_registry()
        for run in reversed(registry):
            if run.get("status") == "running":
                return run
        return None
"""
        new_lines.append(registry_code)
        continue
    
    new_lines.append(line)

with open(file_path, "w") as f:
    f.writelines(new_lines)
