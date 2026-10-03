#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/lzm/SARAF
PYTHON=/home/lzm/timellm-venv/bin/python
RESULTS="$ROOT/saraf_results.csv"
ERRORS="$ROOT/logs/errors.log"
mkdir -p "$ROOT/logs"
touch "$ERRORS"

run_one() {
  local dataset="$1"
  local pred_len="$2"
  local gpu="$3"
  local freq="$4"
  local model_id="$dataset"_sl96_pl"$pred_len"
  local log="$ROOT/logs/$model_id.log"

  echo "[$(date '+%F %T')] START dataset=$dataset seq_len=96 pred_len=$pred_len gpu=$gpu" | tee -a "$log"
  if "$PYTHON" -u "$ROOT/run.py" \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path "$ROOT/data/ETT/" \
      --data_path "$dataset.csv" \
      --model_id "$model_id" \
      --model SARAF \
      --data "$dataset" \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len "$pred_len" \
      --learning_rate 0.001 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --freq "$freq" \
      --des SARAF_seq96 \
      --seed 2021 \
      --itr 1 \
      --topm 20 \
      --time_aware_weight 0.5 \
      --num_workers 4 \
      --batch_size 8 \
      --train_epochs 10 \
      --patience 10 \
      --gpu "$gpu" >> "$log" 2>&1; then
    local metrics mse mae
    metrics=$(grep 'mse:.*mae:' "$log" | tail -n 1 || true)
    mse=$(printf '%s\n' "$metrics" | sed -nE 's/.*mse:([^,]+), mae:.*/\1/p')
    mae=$(printf '%s\n' "$metrics" | sed -nE 's/.*mae:([^,]+), rmse:.*/\1/p')
    if [[ -n "$mse" && -n "$mae" ]]; then
      (
        flock -x 9
        printf 'SARAF,%s,96,%s,%s,%s\n' "$dataset" "$pred_len" "$mse" "$mae" >> "$RESULTS"
      ) 9>"$ROOT/.saraf_results.lock"
      echo "[$(date '+%F %T')] DONE dataset=$dataset pred_len=$pred_len mse=$mse mae=$mae" | tee -a "$log"
    else
      echo "[$(date '+%F %T')] ERROR metrics not found: dataset=$dataset pred_len=$pred_len log=$log" | tee -a "$ERRORS"
    fi
  else
    echo "[$(date '+%F %T')] ERROR training failed: dataset=$dataset pred_len=$pred_len log=$log" | tee -a "$ERRORS"
  fi
}

gpu="$1"
shift
for dataset in "$@"; do
  case "$dataset" in
    ETTh1|ETTh2) freq=h ;;
    ETTm1|ETTm2) freq=t ;;
    *) echo "Unknown dataset: $dataset" >&2; exit 2 ;;
  esac
  for pred_len in 96 192 336 720; do
    run_one "$dataset" "$pred_len" "$gpu" "$freq"
  done
done

echo "[$(date '+%F %T')] QUEUE COMPLETE gpu=$gpu datasets=$*" | tee -a "$ROOT/logs/queue_complete.log"