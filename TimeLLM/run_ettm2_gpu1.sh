#!/usr/bin/env bash
set -u

ROOT=/home/lzm/Time-LLM
PY=/home/lzm/timellm-venv/bin/python
RESULT_DIR="$ROOT/ett_experiments"
CSV="$RESULT_DIR/timellm_ett_results.csv"
CSV_LOCK="$RESULT_DIR/timellm_ett_results.csv.lock"
PATCHED_RUN_MAIN="$ROOT/run_main.py"
GPU="${GPU:-1}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LLM_LAYERS="${LLM_LAYERS:-6}"
MIN_FREE_MB="${MIN_FREE_MB:-20000}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "$RESULT_DIR/logs" "$RESULT_DIR/locks"
cd "$ROOT"

if [[ ! -s "$CSV" ]]; then
  echo 'model,dataset,seq_len,pred_len,mse,mae' > "$CSV"
fi

already_done() {
  local dataset="$1"
  local pred_len="$2"
  awk -F, -v d="$dataset" -v p="$pred_len" \
    'NR > 1 && $1 == "TimeLLM-GPT2" && $2 == d && $3 == 96 && $4 == p { found=1 } END { exit !found }' "$CSV"
}

wait_for_gpu() {
  local free_mb
  while true; do
    free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
    if [[ "$free_mb" =~ ^[0-9]+$ ]] && (( free_mb >= MIN_FREE_MB )); then
      echo "GPU $GPU is ready with ${free_mb} MiB free at $(date -Is)"
      return 0
    fi
    echo "Waiting for GPU $GPU: ${free_mb:-unknown} MiB free, need ${MIN_FREE_MB} MiB at $(date -Is)"
    sleep 300
  done
}

run_one() {
  local dataset="$1"
  local pred_len="$2"
  local freq=t data_path log_file metrics status lock_dir

  if already_done "$dataset" "$pred_len"; then
    echo "Skipping completed run: $dataset pred_len=$pred_len"
    return 0
  fi

  lock_dir="$RESULT_DIR/locks/${dataset}_96_${pred_len}.running"
  if ! mkdir "$lock_dir" 2>/dev/null; then
    echo "Skipping already-running run: $dataset pred_len=$pred_len"
    return 0
  fi
  trap 'rmdir "$lock_dir" 2>/dev/null || true' RETURN

  data_path="${dataset}.csv"
  log_file="$RESULT_DIR/logs/${dataset}_96_${pred_len}.log"

  wait_for_gpu
  echo "Starting $dataset seq_len=96 pred_len=$pred_len on GPU=$GPU at $(date -Is)"
  CUDA_VISIBLE_DEVICES="$GPU" USE_DEEPSPEED=0 "$PY" -u "$PATCHED_RUN_MAIN" \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path "$data_path" \
    --model_id "${dataset}_96_${pred_len}" \
    --model TimeLLM \
    --data "$dataset" \
    --features M \
    --freq "$freq" \
    --seq_len 96 \
    --label_len 48 \
    --pred_len "$pred_len" \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des Exp \
    --itr 1 \
    --d_model 32 \
    --d_ff 128 \
    --batch_size "$BATCH_SIZE" \
    --num_workers 4 \
    --learning_rate 0.01 \
    --llm_model GPT2 \
    --llm_dim 768 \
    --llm_layers "$LLM_LAYERS" \
    --train_epochs "$EPOCHS" \
    --patience 10 \
    --model_comment TimeLLM-GPT2 \
    2>&1 | tee "$log_file"
  status=${PIPESTATUS[0]}

  if [[ "$status" -ne 0 ]]; then
    echo "Run failed: $dataset pred_len=$pred_len (status=$status)" >&2
    return "$status"
  fi

  metrics=$("$PY" - "$log_file" <<'PY'
import re
import sys
pattern = re.compile(r"Vali Loss: ([0-9.eE+-]+) Test Loss: ([0-9.eE+-]+) MAE Loss: ([0-9.eE+-]+)")
rows = []
with open(sys.argv[1], encoding="utf-8", errors="replace") as handle:
    for line in handle:
        match = pattern.search(line)
        if match:
            rows.append(tuple(map(float, match.groups())))
if not rows:
    raise SystemExit("No metrics found in log")
best = min(rows, key=lambda row: row[0])
print(f"{best[1]:.7f},{best[2]:.7f}")
PY
  ) || return 1

  (
    flock 9
    if already_done "$dataset" "$pred_len"; then
      echo "Skipping CSV append because completed by another runner: $dataset pred_len=$pred_len"
    else
      echo "TimeLLM-GPT2,$dataset,96,$pred_len,$metrics" >> "$CSV"
      echo "Completed $dataset pred_len=$pred_len: mse,mae=$metrics"
    fi
  ) 9>"$CSV_LOCK"
}

for pred_len in 96 192 336 720; do
  run_one ETTm2 "$pred_len" || exit $?
done

echo "All ETTm2 GPU1 experiments completed at $(date -Is)"
