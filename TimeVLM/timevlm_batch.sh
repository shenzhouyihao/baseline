#!/usr/bin/env bash
set -euo pipefail
source ~/timellm-venv/bin/activate
export TOKENIZERS_PARALLELISM=false
cd ~/work/ICML25-TimeVLM
mkdir -p logs
final_csv=ett_timevlm_results.csv
g0_csv=ett_timevlm_gpu0.csv
g1_csv=ett_timevlm_gpu1.csv
printf 'model,dataset,seq_len,pred_len,mse,mae\n' > "$g0_csv"
printf 'model,dataset,seq_len,pred_len,mse,mae\n' > "$g1_csv"
run_one() {
  local gpu="$1"
  local group_csv="$2"
  local dataset="$3"
  local pred_len="$4"
  local d_model="$5"
  local use_mem_gate="$6"
  local periodicity="$7"
  local dropout="$8"
  local n_vars=7
  local log_file="logs/${dataset}_sl96_pl${pred_len}_gpu${gpu}.log"
  CUDA_VISIBLE_DEVICES="$gpu" python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path /home/lzm/Time-LLM/dataset/ETT-small/ \
    --data_path "${dataset}.csv" \
    --model_id "${dataset}_96_${pred_len}" \
    --model TimeVLM \
    --data "${dataset}" \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len "${pred_len}" \
    --d_model "${d_model}" \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in "$n_vars" \
    --dec_in "$n_vars" \
    --c_out "$n_vars" \
    --des Exp \
    --itr 1 \
    --gpu 0 \
    --use_amp \
    --train_epochs 15 \
    --image_size 56 \
    --norm_const 0.4 \
    --periodicity "$periodicity" \
    --three_channel_image True \
    --finetune_vlm False \
    --batch_size 32 \
    --learning_rate 0.001 \
    --num_workers 32 \
    --vlm_type clip \
    --use_mem_gate "$use_mem_gate" \
    --dropout "$dropout" \
    --percent 1 >"$log_file" 2>&1
  local line mse mae
  line=$(grep -E 'mse: ' "$log_file" | tail -1 || true)
  if [ -z "$line" ]; then
    echo "missing metrics for ${dataset} ${pred_len} gpu${gpu}" >&2
    tail -20 "$log_file" >&2 || true
    exit 1
  fi
  mse=$(printf '%s' "$line" | sed -E 's/.*mse: ([^,]+), mae: ([^,]+),.*/\1/')
  mae=$(printf '%s' "$line" | sed -E 's/.*mse: ([^,]+), mae: ([^,]+),.*/\2/')
  printf 'TimeVLM,%s,96,%s,%s,%s\n' "$dataset" "$pred_len" "$mse" "$mae" >> "$group_csv"
}
run_gpu0() {
  run_one 0 "$g0_csv" ETTh1 96 32 False 24 0.1
  run_one 0 "$g0_csv" ETTh1 192 32 False 24 0.1
  run_one 0 "$g0_csv" ETTh1 336 64 True 24 0.1
  run_one 0 "$g0_csv" ETTh1 720 256 True 24 0.3
  run_one 0 "$g0_csv" ETTh2 96 64 False 24 0.2
  run_one 0 "$g0_csv" ETTh2 192 64 False 24 0.3
  run_one 0 "$g0_csv" ETTh2 336 128 False 24 0.3
  run_one 0 "$g0_csv" ETTh2 720 32 False 24 0.3
}
run_gpu1() {
  run_one 1 "$g1_csv" ETTm1 96 64 True 96 0.2
  run_one 1 "$g1_csv" ETTm1 192 32 True 96 0.2
  run_one 1 "$g1_csv" ETTm1 336 128 True 96 0.3
  run_one 1 "$g1_csv" ETTm1 720 32 True 96 0.2
  run_one 1 "$g1_csv" ETTm2 96 32 True 96 0.2
  run_one 1 "$g1_csv" ETTm2 192 32 True 96 0.2
  run_one 1 "$g1_csv" ETTm2 336 32 True 96 0.2
  run_one 1 "$g1_csv" ETTm2 720 32 True 96 0.2
}
run_gpu0 &
pid0=$!
run_gpu1 &
pid1=$!
wait "$pid0"
wait "$pid1"
{
  printf 'model,dataset,seq_len,pred_len,mse,mae\n'
  tail -n +2 "$g0_csv"
  tail -n +2 "$g1_csv"
} > "$final_csv"
echo "DONE $final_csv"
