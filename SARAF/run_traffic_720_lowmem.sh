#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/lzm/SARAF
PYTHON=/home/lzm/timellm-venv/bin/python
RESULTS="$ROOT/saraf_weather_electricity_exchange_traffic_seq96_results.csv"
LOCK="$ROOT/.saraf_results.lock"
run_one() {
  dataset="$1"; channels="$2"; freq="$3"; gpu="$4"; pred="$5"; stride="$6"
  awk -F, -v d="$dataset" -v p="$pred" 'NR>1 && $1=="SARAF" && $2==d && $3==96 && $4==p {ok=1} END{exit !ok}' "$RESULTS" && return
  log="$ROOT/logs/${dataset}_sl96_pl${pred}_lowmem_gpu${gpu}.log"
  echo "START dataset=$dataset pred_len=$pred gpu=$gpu eval_stride=$stride" >> "$log"
  if CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$ROOT/run.py" --task_name long_term_forecast --is_training 1 \
    --root_path "$ROOT/data/$dataset/" --data_path "$dataset.csv" --model_id "${dataset}_sl96_pl${pred}_SARAF_lowmem" \
    --model SARAF --data "$dataset" --features M --seq_len 96 --label_len 48 --pred_len "$pred" \
    --learning_rate 0.001 --enc_in "$channels" --dec_in "$channels" --c_out "$channels" --freq "$freq" \
    --des SARAF_seq96 --seed 2021 --itr 1 --topm 20 --time_aware_weight 0.5 --num_workers 0 \
    --batch_size 1 --train_epochs 10 --patience 10 --gpu 0 --eval_stride "$stride" >> "$log" 2>&1; then
    metrics=$(grep 'mse:.*mae:' "$log" | tail -n 1 || true)
    mse=$(printf '%s\n' "$metrics" | sed -nE 's/.*mse:([^,]+), mae:.*/\1/p')
    mae=$(printf '%s\n' "$metrics" | sed -nE 's/.*mse:[^,]+, mae:([^,]+),.*/\1/p')
    if [[ -n "$mse" && -n "$mae" ]]; then
      ( flock -x 9; printf 'SARAF,%s,96,%s,%.6f,%.6f\n' "$dataset" "$pred" "$mse" "$mae" >> "$RESULTS" ) 9>"$LOCK"
      echo "DONE dataset=$dataset pred_len=$pred mse=$mse mae=$mae" >> "$log"
    fi
  else
    echo "ERROR dataset=$dataset pred_len=$pred" >> "$log"
  fi
}
run_one traffic 862 h 1 720 100
