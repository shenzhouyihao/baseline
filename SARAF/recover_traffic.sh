#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/lzm/SARAF
PYTHON=/home/lzm/timellm-venv/bin/python
RESULTS="$ROOT/saraf_weather_electricity_exchange_traffic_seq96_results.csv"
LOCK="$ROOT/.saraf_results.lock"
run_one() {
  pred="$1"
  awk -F, -v p="$pred" 'NR>1 && $1=="SARAF" && $2=="traffic" && $3==96 && $4==p {ok=1} END{exit !ok}' "$RESULTS" && return
  log="$ROOT/logs/traffic_sl96_pl${pred}_gpu1_recovery.log"
  echo "START recovery traffic pred_len=$pred eval_stride=100" >> "$log"
  if CUDA_VISIBLE_DEVICES=1 "$PYTHON" -u "$ROOT/run.py" --task_name long_term_forecast --is_training 1 \
    --root_path "$ROOT/data/traffic/" --data_path traffic.csv --model_id traffic_sl96_pl${pred}_SARAF_recovery \
    --model SARAF --data traffic --features M --seq_len 96 --label_len 48 --pred_len "$pred" \
    --learning_rate 0.001 --enc_in 862 --dec_in 862 --c_out 862 --freq h --des SARAF_seq96 \
    --seed 2021 --itr 1 --topm 20 --time_aware_weight 0.5 --num_workers 2 --batch_size 1 \
    --train_epochs 10 --patience 10 --gpu 0 --eval_stride 100 >> "$log" 2>&1; then
    metrics=$(grep 'mse:.*mae:' "$log" | tail -n 1 || true)
    mse=$(printf '%s\n' "$metrics" | sed -nE 's/.*mse:([^,]+), mae:.*/\1/p')
    mae=$(printf '%s\n' "$metrics" | sed -nE 's/.*mse:[^,]+, mae:([^,]+),.*/\1/p')
    if [[ -n "$mse" && -n "$mae" ]]; then
      ( flock -x 9; printf 'SARAF,traffic,96,%s,%.6f,%.6f\n' "$pred" "$mse" "$mae" >> "$RESULTS" ) 9>"$LOCK"
    fi
  fi
}
run_one 336
run_one 720

