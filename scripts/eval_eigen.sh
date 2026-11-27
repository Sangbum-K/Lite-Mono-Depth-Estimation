#!/usr/bin/env bash
# Eigen test (697) evaluation of one checkpoint folder, monodepth2 protocol
# (median scaling, depth cap 80m, Garg crop). Needs splits/eigen/gt_depths.npz
# (vendor/Lite-Mono/export_gt_depth.py --split eigen).
# Usage: scripts/eval_eigen.sh <weights_folder> <log_file> [extra evaluate_depth.py args...]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/configs/paths.env"
WEIGHTS="$(realpath "${1:?usage: $0 <weights_folder> <log_file>}")"
LOG="$(realpath -m "${2:?usage: $0 <weights_folder> <log_file>}")"; shift 2
mkdir -p "$(dirname "$LOG")"

source "$ROOT/.venv/bin/activate"
cd "$ROOT/vendor/Lite-Mono"

python evaluate_depth.py \
  --data_path "$KITTI_ROOT" --png \
  --model lite-mono \
  --eval_split eigen \
  --num_workers 6 \
  --load_weights_folder "$WEIGHTS" \
  "$@" > "$LOG" 2>&1
grep -E "abs_rel|^&" "$LOG"
