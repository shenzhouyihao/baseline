#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lzm/work/TS-RAG/TS-RAG
PY=/home/lzm/timellm-venv/bin/python
CSV="$ROOT/tsrag_results.csv"
LOG_DIR="$ROOT/logs"
BASE_MODEL="$ROOT/checkpoints/base"
CHECKPOINT="$ROOT/checkpoints/chronos-bolt/best.pth"
DATA_ROOT=/home/lzm/Time-LLM/dataset/ETT-small
RETRIEVE_ROOT="$ROOT/retrieval_database"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64}"

mkdir -p "$LOG_DIR" "$BASE_MODEL"

if [[ ! -f "$CSV" ]]; then
  printf 'model,dataset,seq_len,pred_len,mse,mae\n' > "$CSV"
fi

run_one() {
  local gpu="$1"
  local dataset="$2"
  local pred_len="$3"
  local data_kind="$4"
  local freq="$5"
  local meta_freq="$6"
  local log_file="$LOG_DIR/${dataset}_96_${pred_len}_gpu${gpu}.log"

  echo "[$(date -Is)] start dataset=${dataset} pred_len=${pred_len} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$ROOT/zeroshot.py" \
    --model_id "${dataset}_zeroshot_96_pred_${pred_len}_retrieve_${pred_len}" \
    --root_path "$DATA_ROOT" \
    --data_path "${dataset}.csv" \
    --data "$data_kind" \
    --features M \
    --freq "$freq" \
    --percent 100 \
    --seq_len 96 \
    --label_len 0 \
    --pred_len "$pred_len" \
    --batch_size 128 \
    --num_workers 4 \
    --model ChronosBoltRetrieve \
    --gpu_loc 0 \
    --checkpoint_model_path "$CHECKPOINT" \
    --pretrained_model_path "$BASE_MODEL" \
    --retrieval_database_dir "$RETRIEVE_ROOT" \
    --metadata_frequency "$meta_freq" \
    --metadata_database_name "$dataset" \
    --embedding_model_type chronos \
    --augment_mode moe \
    --top_k 10 \
    --save_file_name "tsrag_rolling_${dataset}.txt" \
    --csv_path "$CSV" \
    2>&1 | tee "$log_file"
}

run_group() {
  local gpu="$1"
  shift
  local dataset pred_len

  for dataset in "$@"; do
    for pred_len in 96 192 336 720; do
      if [[ "$dataset" == ETTh1 || "$dataset" == ETTh2 ]]; then
        run_one "$gpu" "$dataset" "$pred_len" ett_h_retrieve h hour
      else
        run_one "$gpu" "$dataset" "$pred_len" ett_m_retrieve t minute
      fi
    done
  done
}

run_group 0 ETTh1 ETTh2 &
pid0=$!
run_group 1 ETTm1 ETTm2 &
pid1=$!
wait "$pid0" "$pid1"

echo "[$(date -Is)] all runs finished"
