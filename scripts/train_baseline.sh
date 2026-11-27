#!/usr/bin/env bash
# Phase 3 baseline: original Lite-Mono, 640x192, ImageNet-initialised encoder.
# Hyper-parameters follow the official README training command.
# Usage: scripts/train_baseline.sh <run_id> [extra train.py args...]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/configs/paths.env"
RUN_ID="${1:?usage: $0 <run_id> [extra args]}"; shift || true
OUT="$OUTPUT_ROOT/$RUN_ID"
mkdir -p "$OUT"

source "$ROOT/.venv/bin/activate"
cd "$ROOT/vendor/Lite-Mono"   # train.py reads splits/ relative to itself

# num_workers=6: 12 hit host-RAM OOM on the 15GB workstation (val loader spawns
# its own workers too, so the real worker count is 2x this value).
exec python -u train.py \
  --data_path "$KITTI_ROOT" --png \
  --split eigen_zhou \
  --model lite-mono \
  --height 192 --width 640 \
  --batch_size 12 \
  --num_epochs 30 \
  --lr 0.0001 5e-6 31 0.0001 1e-5 31 \
  --mypretrain "$PRETRAINED_ROOT/imagenet/lite-mono-pretrain.pth" \
  --num_workers 6 \
  --log_dir "$OUTPUT_ROOT" --model_name "$RUN_ID" \
  "$@" 2>&1 | tee -a "$OUT/train.log"
