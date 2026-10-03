#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lzm/work/TS-RAG/TS-RAG
PY=/home/lzm/timellm-venv/bin/python
DATA_ROOT=/home/lzm/work/TS-RAG/datasets
RETRIEVE_ROOT="$ROOT/retrieval_database"
RESULTS="$ROOT/tsrag_96_benchmark_results.csv"
LOG_DIR="$ROOT/logs/tsrag_96_benchmark"
CHECKPOINT="$ROOT/checkpoints/chronos-bolt/best.pth"
BASE_MODEL="$ROOT/checkpoints/base"
DATASET_REPO=https://hf-mirror.com/datasets/thuml/Time-Series-Library/resolve/main

export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

mkdir -p "$DATA_ROOT" "$RETRIEVE_ROOT" "$LOG_DIR"

download_dataset() {
  local dataset="$1"
  local directory="$DATA_ROOT/$dataset"
  local destination="$directory/$dataset.csv"
  mkdir -p "$directory"
  if [[ ! -s "$destination" ]]; then
    echo "[$(date -Is)] downloading $dataset"
    curl -4 -fL --retry 8 --retry-delay 5 -C - \
      "$DATASET_REPO/$dataset/$dataset.csv?download=true" -o "$destination"
  fi
  "$PY" -c "import pandas as pd; df=pd.read_csv('$destination'); assert df.shape[0] > 1000 and df.shape[1] > 1; assert df.columns[0] == 'date'; print('$dataset:', df.shape)"
}

for dataset in weather electricity exchange_rate traffic; do
  download_dataset "$dataset"
done

if [[ ! -f "$RESULTS" ]]; then
  printf 'model,dataset,seq_len,pred_len,mse,mae\n' > "$RESULTS"
fi

is_complete() {
  local dataset="$1"
  local pred_len="$2"
  local csv_dataset="$dataset"
  if [[ "$dataset" == "exchange_rate" ]]; then
    csv_dataset="exchange"
  fi
  awk -F, -v dataset="$csv_dataset" -v pred_len="$pred_len" \
    'NR > 1 && $1 == "ChronosBoltRetrieve" && $2 == dataset && $3 == "96" && $4 == pred_len { found=1 } END { exit !found }' \
    "$RESULTS"
}

run_one() {
  local gpu="$1"
  local dataset="$2"
  local pred_len="$3"
  local frequency="$4"
  local batch_size="$5"
  local log_file="$LOG_DIR/${dataset}_seq96_pred${pred_len}_gpu${gpu}.log"

  if is_complete "$dataset" "$pred_len"; then
    echo "[$(date -Is)] skip completed dataset=$dataset pred_len=$pred_len"
    return
  fi

  echo "[$(date -Is)] start dataset=$dataset seq_len=96 pred_len=$pred_len gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$ROOT/zeroshot.py" \
    --model_id "${dataset}_zeroshot_96_pred_${pred_len}_retrieve_${pred_len}" \
    --root_path "$DATA_ROOT/$dataset" \
    --data_path "$dataset.csv" \
    --data custom_retrieve \
    --features M \
    --freq h \
    --percent 100 \
    --seq_len 96 \
    --label_len 0 \
    --pred_len "$pred_len" \
    --lookback_length 96 \
    --batch_size "$batch_size" \
    --num_workers 0 \
    --model ChronosBoltRetrieve \
    --gpu_loc 0 \
    --checkpoint_model_path "$CHECKPOINT" \
    --pretrained_model_path "$BASE_MODEL" \
    --retrieval_database_dir "$RETRIEVE_ROOT" \
    --metadata_frequency "$frequency" \
    --metadata_database_name "$dataset" \
    --embedding_model_type chronos \
    --augment_mode moe \
    --top_k 10 \
    --streaming_self_retrieval \
    --eval_stride 100 \
    --save_file_name "tsrag_96_${dataset}.txt" \
    --csv_path "$RESULTS" \
    > "$log_file" 2>&1
  echo "[$(date -Is)] complete dataset=$dataset seq_len=96 pred_len=$pred_len gpu=$gpu"
}

run_dataset() {
  local gpu="$1"
  local dataset="$2"
  local frequency="$3"
  local batch_size="$4"
  local pred_len
  for pred_len in 96 192 336 720; do
    run_one "$gpu" "$dataset" "$pred_len" "$frequency" "$batch_size"
  done
}

run_gpu0() {
  run_dataset 0 weather 10minutes 128
  run_dataset 0 electricity hour 8
}

run_gpu1() {
  run_dataset 1 exchange_rate hour 128
  run_dataset 1 traffic hour 2
}

if [[ "${TSRAG_GROUP:-all}" == "all" || "${TSRAG_GROUP:-}" == "gpu0" ]]; then
  run_gpu0 &
  pid0=$!
fi
if [[ "${TSRAG_GROUP:-all}" == "all" || "${TSRAG_GROUP:-}" == "gpu1" ]]; then
  run_gpu1 &
  pid1=$!
fi
if [[ -n "${pid0:-}" ]]; then
  wait "$pid0"
fi
if [[ -n "${pid1:-}" ]]; then
  wait "$pid1"
fi
echo "[$(date -Is)] all requested experiments completed"
