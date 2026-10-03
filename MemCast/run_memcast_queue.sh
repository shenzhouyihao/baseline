#!/usr/bin/env bash
set -eu
ROOT=/home/lzm/work/MemCast
GPU=$1
DATASETS=$2
LOG=$ROOT/logs/memcast_gpu${GPU}.log
mkdir -p "$ROOT/logs" "$ROOT/results"
cd "$ROOT"
source "$ROOT/.memcast.env"
export CUDA_VISIBLE_DEVICES="$GPU"
export MEMCAST_LOW_MEMORY=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MEMCAST_DATASETS="$DATASETS"
export MEMCAST_PRED_LENS="96,192,336,720"
export MEMCAST_NUM_SAMPLES=30
export MEMCAST_CSV="$ROOT/results/memcast_eight_datasets.csv"
export MEMCAST_FAILURE_CSV="$ROOT/results/memcast_opus46_failed_samples.csv"
echo "START MemCast gpu=$GPU datasets=$DATASETS samples=$MEMCAST_NUM_SAMPLES $(date -Is)" | tee -a "$LOG"
/home/lzm/timellm-venv/bin/python -u memcast_experiment_runner.py >> "$LOG" 2>&1
echo "DONE MemCast gpu=$GPU $(date -Is)" | tee -a "$LOG"
