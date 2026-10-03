#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lzm/Time-LLM
PY=/home/lzm/timellm-venv/bin/python
OUT="$ROOT/ett_recovered"
CKPT="$ROOT/ett_recovered_checkpoints"
CSV="$OUT/timellm_ett_recovered_results.csv"
GPU="${GPU:-1}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LLM_LAYERS="${LLM_LAYERS:-6}"
MIN_FREE_MB="${MIN_FREE_MB:-12000}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "$OUT/logs" "$CKPT"
cd "$ROOT"
if [[ ! -s "$CSV" ]]; then
  echo 'model,dataset,seq_len,pred_len,mse,mae,checkpoint' > "$CSV"
fi

wait_for_gpu() {
  while true; do
    free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
    if [[ "$free_mb" =~ ^[0-9]+$ ]] && (( free_mb >= MIN_FREE_MB )); then
      echo "GPU $GPU ready: ${free_mb} MiB free at $(date -Is)"
      return
    fi
    echo "Waiting for GPU $GPU: ${free_mb:-unknown} MiB free at $(date -Is)"
    sleep 180
  done
}

for dataset in ETTh1 ETTh2 ETTm1 ETTm2; do
  case "$dataset" in
    ETTh1|ETTh2) freq=h ;;
    ETTm1|ETTm2) freq=t ;;
  esac
  for pred_len in 96 192 336 720; do
    log_file="$OUT/logs/${dataset}_96_${pred_len}.log"
    echo "Starting $dataset pred_len=$pred_len at $(date -Is)"
    wait_for_gpu
    CUDA_VISIBLE_DEVICES="$GPU" USE_DEEPSPEED=0 "$PY" -u "$ROOT/run_main.py" \
      --task_name long_term_forecast --is_training 1 \
      --root_path ./dataset/ETT-small/ --data_path "${dataset}.csv" \
      --model_id "${dataset}_96_${pred_len}_recovered" --model TimeLLM --data "$dataset" \
      --features M --freq "$freq" --seq_len 96 --label_len 48 --pred_len "$pred_len" \
      --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des Recovered --itr 1 \
      --d_model 32 --d_ff 128 --batch_size "$BATCH_SIZE" --num_workers 4 \
      --learning_rate 0.01 --llm_model GPT2 --llm_dim 768 --llm_layers "$LLM_LAYERS" \
      --train_epochs "$EPOCHS" --patience 10 --checkpoints "$CKPT" \
      --model_comment TimeLLM-GPT2-Recovered 2>&1 | tee "$log_file"
    "$PY" - "$log_file" "$CSV" "$dataset" "$pred_len" "$CKPT" <<'PY'
import re, sys
log_file, csv_file, dataset, pred_len, ckpt_root = sys.argv[1:]
pattern = re.compile(r"BEST_CHECKPOINT Test Loss: ([0-9.eE+-]+) MAE Loss: ([0-9.eE+-]+)")
matches = []
with open(log_file, encoding="utf-8", errors="replace") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            matches.append((float(m.group(1)), float(m.group(2))))
if len(matches) != 1:
    raise SystemExit(f"expected one best-checkpoint metric, got {len(matches)}")
from pathlib import Path
dirs = [p for p in Path(ckpt_root).glob(f"*{dataset}_96_{pred_len}_recovered*") if (p / "checkpoint").is_file()]
if len(dirs) != 1:
    raise SystemExit(f"expected one checkpoint for {dataset}/{pred_len}, got {len(dirs)}")
mse, mae = matches[0]
with open(csv_file, "a", encoding="utf-8") as f:
    f.write(f"TimeLLM-GPT2,{dataset},96,{pred_len},{mse:.7f},{mae:.7f},{dirs[0] / 'checkpoint'}\n")
print(f"RECOVERED_ROW,{dataset},{pred_len},{mse:.7f},{mae:.7f},{dirs[0] / 'checkpoint'}")
PY
  done
done
echo "All recovered ETT experiments completed at $(date -Is)"

