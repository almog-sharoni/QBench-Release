# Execution Rules

1. **Docker Execution**: All project code, python scripts, and tests MUST be executed inside the `qbench` Docker container.
   - **Do not** run `python` or `pytest` directly on the host.
   - Use `docker exec -it qbench <command>` to run commands.
   - Example: `docker exec -it qbench python runspace/run_interactive.py`
   - Exception: Docker management commands (build, run, ps) are run on the host.
