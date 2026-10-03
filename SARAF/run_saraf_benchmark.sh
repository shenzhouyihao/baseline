#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/lzm/SARAF
PYTHON=/home/lzm/timellm-venv/bin/python
RESULTS="$ROOT/saraf_weather_electricity_exchange_traffic_seq96_results.csv"
LOCK="$ROOT/.saraf_results.lock"
mkdir -p "$ROOT/logs"
if [[ ! -f "$RESULTS" ]]; then printf 'model,dataset,seq_len,pred_len,mse,mae\n' > "$RESULTS"; fi
is_done() { awk -F, -v d="$1" -v p="$2" 'NR > 1 && $1 == "SARAF" && $2 == d && $3 == 96 && $4 == p {ok=1} END {exit !ok}' "$RESULTS"; }
run_one() {
  dataset="$1"; pred="$2"; gpu="$3"; freq="$4"; channels="$5"; batch="$6"; stride="$7"
  is_done "$dataset" "$pred" && return
  log="$ROOT/logs/"$dataset"_sl96_pl"$pred"_gpu"$gpu".log"
  echo "START dataset=$dataset pred_len=$pred gpu=$gpu eval_stride=$stride" | tee -a "$log"
  if CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$ROOT/run.py" --task_name long_term_forecast --is_training 1 \
    --root_path "$ROOT/data/$dataset/" --data_path "$dataset.csv" --model_id "$dataset"_sl96_pl"$pred"_SARAF \
    --model SARAF --data "$dataset" --features M --seq_len 96 --label_len 48 --pred_len "$pred" \
    --learning_rate 0.001 --enc_in "$channels" --dec_in "$channels" --c_out "$channels" --freq "$freq" \
    --des SARAF_seq96 --seed 2021 --itr 1 --topm 20 --time_aware_weight 0.5 --num_workers 4 \
    --batch_size "$batch" --train_epochs 10 --patience 10 --gpu 0 --eval_stride "$stride" >> "$log" 2>&1; then
    metrics=$(grep 'mse:.*mae:' "$log" | tail -n 1 || true)
    mse=$(printf '%s\n' "$metrics" | sed -nE 's/.*mse:([^,]+), mae:.*/\1/p')
    mae=$(printf '%s\n' "$metrics" | sed -nE 's/.*mse:[^,]+, mae:([^,]+),.*/\1/p')
    if [[ -n "$mse" && -n "$mae" ]]; then
      ( flock -x 9; printf 'SARAF,%s,96,%s,%.6f,%.6f\n' "$dataset" "$pred" "$mse" "$mae" >> "$RESULTS" ) 9>"$LOCK"
      echo "DONE dataset=$dataset pred_len=$pred mse=$mse mae=$mae" | tee -a "$log"
    else
      echo "ERROR metrics dataset=$dataset pred=$pred" >> "$ROOT/logs/saraf_errors.log"
    fi
  else
    echo "ERROR run dataset=$dataset pred=$pred" >> "$ROOT/logs/saraf_errors.log"
  fi
}
run_gpu0() { for p in 96 192 336 720; do run_one weather "$p" 0 10min 21 8 1; done; for p in 96 192 336 720; do run_one electricity "$p" 0 h 321 4 100; done; }
run_gpu1() { for p in 96 192 336 720; do run_one exchange_rate "$p" 1 h 8 32 1; done; for p in 96 192 336 720; do run_one traffic "$p" 1 h 862 2 100; done; }
run_gpu0 & a=$!
run_gpu1 & b=$!
wait "$a"
wait "$b"
echo "QUEUE COMPLETE" >> "$ROOT/logs/saraf_queue_complete.log"
