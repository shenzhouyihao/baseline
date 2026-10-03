#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lzm/Time-LLM
PY=/home/lzm/timellm-venv/bin/python
OUT="$ROOT/ett_recovered_v2"
CKPT="$ROOT/ett_recovered_checkpoints_v2"
CSV="$OUT/timellm_ett_recovered_results.csv"
mkdir -p "$OUT/logs" "$CKPT"
cd "$ROOT"

for pred_len in 96 192 336 720; do
  log_file="$OUT/logs/ETTm2_96_${pred_len}.log"
  if [[ -s "$CSV" ]] && grep -q ",ETTm2,96,$pred_len," "$CSV"; then
    echo "Skipping completed ETTm2 pred_len=$pred_len"
    continue
  fi
  echo "Starting ETTm2 pred_len=$pred_len at $(date -Is)"
  while true; do
    free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')
    if [[ "$free_mb" =~ ^[0-9]+$ ]] && (( free_mb >= 10000 )); then
      break
    fi
    echo "Waiting for GPU 0: ${free_mb:-unknown} MiB free"
    sleep 120
  done
  CUDA_VISIBLE_DEVICES=0 USE_DEEPSPEED=0 "$PY" -u "$ROOT/run_main.py" \
    --task_name long_term_forecast --is_training 1 \
    --root_path ./dataset/ETT-small/ --data_path ETTm2.csv \
    --model_id "ETTm2_96_${pred_len}_recovered" --model TimeLLM --data ETTm2 \
    --features M --freq t --seq_len 96 --label_len 48 --pred_len "$pred_len" \
    --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des Recovered --itr 1 \
    --d_model 32 --d_ff 128 --batch_size 8 --num_workers 4 \
    --learning_rate 0.01 --llm_model GPT2 --llm_dim 768 --llm_layers 6 \
    --train_epochs 100 --patience 10 --checkpoints "$CKPT" \
    --model_comment TimeLLM-GPT2-Recovered-v2 2>&1 | tee "$log_file"
  "$PY" - "$log_file" "$CSV" "$pred_len" "$CKPT" <<'PY'
import re, sys
from pathlib import Path
log_file, csv_file, pred_len, ckpt_root = sys.argv[1:]
pattern = re.compile(r"BEST_CHECKPOINT Test Loss: ([0-9.eE+-]+) MAE Loss: ([0-9.eE+-]+)")
matches = []
with open(log_file, encoding="utf-8", errors="replace") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            matches.append((float(m.group(1)), float(m.group(2))))
if len(matches) != 1:
    raise SystemExit(f"expected one best-checkpoint metric, got {len(matches)}")
dirs = [p for p in Path(ckpt_root).glob(f"*ETTm2_96_{pred_len}_recovered*") if (p / "checkpoint").is_file()]
if len(dirs) != 1:
    raise SystemExit(f"expected one checkpoint for ETTm2/{pred_len}, got {len(dirs)}")
mse, mae = matches[0]
with open(csv_file, "a", encoding="utf-8") as f:
    f.write(f"TimeLLM-GPT2,ETTm2,96,{pred_len},{mse:.7f},{mae:.7f},{dirs[0] / 'checkpoint'}\n")
print(f"RECOVERED_ROW,ETTm2,{pred_len},{mse:.7f},{mae:.7f},{dirs[0] / 'checkpoint'}")
PY
done
echo "ETTm2 GPU0 completed at $(date -Is)"

