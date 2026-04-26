#!/usr/bin/env bash
set -euo pipefail

SRC="/data/shared_data/scannet/"
DEST_USER="yarden"
DEST_HOST="132.70.226.91"
DEST_PATH="/data/shared_data/scannet/"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/runspace/outputs/logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/copy_scannet_to_remote_${TS}.log"

echo "[$(date -Iseconds)] Starting rsync ${SRC} -> ${DEST_USER}@${DEST_HOST}:${DEST_PATH}" | tee -a "${LOG_FILE}"
echo "[$(date -Iseconds)] Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"

# -a: archive (preserve perms/times/symlinks)
# -h: human-readable sizes
# -v: verbose
# --info=progress2: overall progress
# --partial: keep partial files for resume
# --append-verify: resume large files, verify on completion
# -e ssh: explicit transport
rsync -ahv \
  --info=progress2 \
  --partial \
  --append-verify \
  -e ssh \
  "${SRC}" \
  "${DEST_USER}@${DEST_HOST}:${DEST_PATH}" \
  2>&1 | tee -a "${LOG_FILE}"

RC=${PIPESTATUS[0]}
echo "[$(date -Iseconds)] rsync exit code: ${RC}" | tee -a "${LOG_FILE}"
exit "${RC}"
