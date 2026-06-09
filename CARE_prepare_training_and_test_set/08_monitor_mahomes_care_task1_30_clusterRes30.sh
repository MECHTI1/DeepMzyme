#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_common_care_task1_30_clusterRes30.sh"
echo "[INFO] WORK_ROOT=$WORK_ROOT"
echo "[INFO] disk:"
df -h "$WORK_ROOT" || true
echo "[INFO] AlphaFill files:"
find "$WORK_ROOT/alphafill" -type f 2>/dev/null | wc -l || true
echo "[INFO] MAHOMES candidate PDBs:"
find "$WORK_ROOT/mahomes_inputs" -maxdepth 2 -type f -name '*.pdb' 2>/dev/null | wc -l || true
echo "[INFO] running CARE/MAHOMES processes:"
ps -eo pid,ppid,stat,etime,pcpu,pmem,cmd | grep -E 'care_prepare.py|run_mahomes_care|MAHOMES|mahomes' | grep -v grep || true
echo "[INFO] latest logs:"
ls -ltr "$LOG_DIR"/*.log 2>/dev/null | tail -10 || true
