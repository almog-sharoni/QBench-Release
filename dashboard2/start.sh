#!/bin/bash
# QBench Dashboard v2 - Start Script
# Usage: ./start.sh          (production mode - builds then serves)
#        ./start.sh dev      (development mode - hot reload)

DIR="$(cd "$(dirname "$0")" && pwd)"

if [ "$1" = "dev" ]; then
  echo "Starting in DEVELOPMENT mode (hot reload)..."
  cd "$DIR" && npm run dev
else
  echo "Building and starting in PRODUCTION mode..."
  cd "$DIR" && npm run build && node server.js
fi
