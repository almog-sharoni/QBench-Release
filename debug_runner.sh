#!/bin/bash

# Check if a script argument is provided
if [ -z "$1" ]; then
    echo "Usage: ./debug_runner.sh <path_to_script> [args...]"
    echo "Example: ./debug_runner.sh runspace/run_interactive.py"
    exit 1
fi

SCRIPT=$1
shift

echo "Starting $SCRIPT in container 'qbench' with debugger waiting on port 5678..."
echo "Please start the 'Python: Attach to Container (QBench)' debug session in VS Code now."

# Run the script inside the container using debugpy
# We use 0.0.0.0 to allow external connections
docker exec -it qbench python -m debugpy --listen 0.0.0.0:5678 --wait-for-client "$SCRIPT" "$@"
