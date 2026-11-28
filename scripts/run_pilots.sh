#!/usr/bin/env bash
# 5-epoch pilots for the intermediate report, run ONE AT A TIME (15GB RAM: never run two trainings
# or a training plus analysis together). Each pilot: train 5 epochs -> Eigen eval of weights_4 ->
# failure analysis (no figures) -> edge offsets. Finished pilots (eval log present) are skipped, so
# the script can simply be started again after a crash. Order: B, A-lite, C, A (cheapest first).
# If training is OOM-killed it is retried once with 2 data workers; if that also fails the script
# stops (it does not run on to the next pilot) so nothing is launched into a memory-starved machine.
# Never edit this file in place while it runs (bash reads it lazily): write a new file and mv it.
#
# Base (baseline rerun) runs only when named: scripts/run_pilots.sh Base
# Usage (from a normal terminal):
#   nohup scripts/run_pilots.sh > outputs/pilots.log 2>&1 &
#   scripts/run_pilots.sh B C          # only some pilots
#   scripts/run_pilots.sh E            # candidate E (Depth Pro 2 decoder, width 24)
# Progress: scripts/pilot_status.py [--watch]  (or tail -f outputs/pilots.log / outputs/<run>/train.log)
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/configs/paths.env"
PY="$ROOT/.venv/bin/python"
WORKERS="${WORKERS:-6}"   # train-loader workers (val loader uses 1); falls back to 2 on OOM
# Data loading is CPU-bound (PNG decode + 4-scale Lanczos resize, ~150 ms/sample/worker). One thread
# per worker stops OpenMP/BLAS thread pools (33 threads/worker) from fighting over the 16 CPU threads.
# Training runs on the GPU, so this changes speed only.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# only one instance at a time, and never next to another training (two trainings = OOM on 15GB)
exec 9>"$OUTPUT_ROOT/.run_pilots.lock"
flock -n 9 || { echo "run_pilots.sh is already running (see scripts/pilot_status.py)"; exit 1; }
if pgrep -f "python -u train.py" > /dev/null; then
  echo "a training (train.py) is already running; start this after it ends (scripts/pilot_status.py)"; exit 1
fi

declare -A RUN ARGS
RUN[B]=20261127_smooth1e-4_e5;     ARGS[B]="--disparity_smoothness 1e-4"
RUN[Alite]=20261127_hires_lite_e5; ARGS[Alite]="--hires_decoder --hires_stem_ch 8 --hires_upsample subpixel"
RUN[C]=20261127_smoothgain10_e5;   ARGS[C]="--smooth_edge_gain 10"
RUN[A]=20261127_hires_decoder_e5;  ARGS[A]="--hires_decoder"
# baseline again with identical settings: run-to-run (training) noise of the 5-epoch protocol
RUN[Base]=20261128_baseline_rerun_e5; ARGS[Base]=""
# candidate E (2026-11-28): Depth Pro 2 / DPT decoder, width 24 (+2.0% MACs, GPU time +6-8%). Runs only when named.
RUN[E]=20261128_dpt24_e5;          ARGS[E]="--dpt_decoder --dpt_dim 24 --dpt_head_ch 16"
if [ $# -gt 0 ]; then ORDER=("$@"); else ORDER=(B Alite C A); fi

for key in "${ORDER[@]}"; do
  run="${RUN[$key]:?unknown pilot $key (B, Alite, C, A, Base, E)}"; args="${ARGS[$key]}"
  out="$OUTPUT_ROOT/$run"; w="$out/models/weights_4"; ana="$out/analysis/weights_4"
  echo "=== $(date '+%F %T') pilot $key: $run ($args)"
  if [ -f "$out/eval/weights_4.log" ] && [ -f "$ana/edges/edges.csv" ]; then echo "already done, skip"; continue; fi
  if [ ! -f "$w/depth.pth" ]; then
    ok=0
    for nw in "$WORKERS" 2; do
      rm -rf "$out"
      echo "$(date '+%F %T') training with $nw data workers, RAM available $(free -g | awk '/Mem:/{print $7}')G (progress: scripts/pilot_status.py)"
      # shellcheck disable=SC2086
      if "$ROOT/scripts/train_baseline.sh" "$run" $args --num_epochs 5 --num_workers "$nw" --val_num_workers 1 > /dev/null 2>&1 \
         && [ -f "$w/depth.pth" ]; then ok=1; break; fi
      echo "$(date '+%F %T') training FAILED with $nw workers (OOM? journalctl -k | grep -i 'killed process')"
      cp "$out/train.log" "$OUTPUT_ROOT/${run}_failed_w${nw}.log" 2>/dev/null
      [ "$nw" = 2 ] && break
      sleep 30
    done
    [ "$ok" = 1 ] || { echo "STOP: $key failed twice. Free RAM (close apps), then run scripts/run_pilots.sh again."; exit 1; }
  fi
  # test time needs only the model-structure flags (all --hires_* / --dpt_*); B/C flags are training-only
  evargs=""; [[ "$args" == --hires_* || "$args" == --dpt_* ]] && evargs="$args"
  for ep in 3 4; do
  w="$out/models/weights_$ep"; ana="$out/analysis/weights_$ep"
  [ -f "$out/eval/weights_$ep.log" ] && [ -f "$ana/edges/edges.csv" ] && continue
  mkdir -p "$ana"
  echo "$(date '+%F %T') evaluating weights_$ep"
  # shellcheck disable=SC2086
  "$ROOT/scripts/eval_eigen.sh" "$w" "$out/eval/weights_$ep.log" $evargs --num_workers "$WORKERS" || echo "eval FAILED"
  echo "$(date '+%F %T') analysing"
  # shellcheck disable=SC2086
  "$PY" "$ROOT/analysis/baseline_failure_analysis.py" --weights "$w" --out "$ana" $evargs --num_workers "$WORKERS" --no_figs > "$ana.log" 2>&1 \
    && "$PY" "$ROOT/analysis/edge_offset.py" --analysis_dir "$ana" > "$ana.edges.log" 2>&1 || echo "analysis FAILED (see $ana.log)"
  done
  echo "$(date '+%F %T') pilot $key done"
done
"$PY" "$ROOT/analysis/pilot_summary.py"
