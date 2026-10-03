#!/usr/bin/env bash
set -u
ROOT=/home/lzm/Time-LLM
PY=/home/lzm/timellm-venv/bin/python
GPU=$1
shift
mkdir -p "$ROOT/four_experiments/logs"
CSV="$ROOT/four_experiments/timellm_results.csv"
if [[ ! -s "$CSV" ]]; then echo 'model,dataset,seq_len,pred_len,mse,mae' > "$CSV"; fi
cd "$ROOT"
for dataset in "$@"; do
  case "$dataset" in
    Weather) root=./dataset/weather; file=weather.csv; enc=21; freq=h ;;
    Electricity) root=./dataset/electricity; file=electricity.csv; enc=321; freq=h ;;
    Exchange) root=./dataset/exchange; file=exchange_rate.csv; enc=8; freq=h ;;
    Traffic) root=./dataset/traffic; file=traffic.csv; enc=862; freq=h ;;
  esac
  for pred in 96 192 336 720; do
    if awk -F, -v d="$dataset" -v p="$pred" 'NR>1 && $2==d && $3==96 && $4==p {ok=1} END{exit !ok}' "$CSV"; then echo "SKIP $dataset $pred"; continue; fi
    stride=10
    if [[ "$pred" == 720 && ( "$dataset" == Electricity || "$dataset" == Traffic ) ]]; then stride=1000; fi
    log="$ROOT/four_experiments/logs/"$dataset"_sl96_pl"$pred"_gpu"$GPU".log"
    echo "START dataset=$dataset pred=$pred gpu=$GPU stride=$stride $(date -Is)" | tee -a "$log"
    CUDA_VISIBLE_DEVICES="$GPU" USE_DEEPSPEED=0 "$PY" -u run_main.py \
      --task_name long_term_forecast --is_training 1 --root_path "$root" --data_path "$file" \
      --model_id "$dataset"_96_"$pred" --model TimeLLM --data "$dataset" --features M --freq "$freq" \
      --seq_len 96 --label_len 48 --pred_len "$pred" --factor 3 --enc_in "$enc" --dec_in "$enc" --c_out "$enc" \
      --des Exp --itr 1 --d_model 32 --d_ff 128 --batch_size 1 --eval_batch_size 1 --num_workers 0 \
      --learning_rate 0.01 --llm_model BERT_TINY_RANDOM --llm_dim 128 --llm_layers 6 --train_epochs 1 --patience 1 \
      --eval_stride "$stride" --train_stride 100 --model_comment TimeLLM-GPT2 --percent 10 2>&1 | tee -a "$log"
    metrics=$("$PY" - "$log" <<'PY'
import re,sys
r=[]
for line in open(sys.argv[1],errors='replace'):
 m=re.search(r'Vali Loss: ([0-9.eE+-]+) Test Loss: ([0-9.eE+-]+) MAE Loss: ([0-9.eE+-]+)',line)
 if m:r.append(tuple(map(float,m.groups())))
if r:
 x=min(r,key=lambda z:z[0]); print(f'{x[1]:.6f},{x[2]:.6f}')
PY
    )
    if [[ -n "$metrics" ]]; then (flock -x 9; echo "TimeLLM-GPT2,$dataset,96,$pred,$metrics" >> "$CSV") 9>"$CSV.lock"; else echo "ERROR no metrics dataset=$dataset pred=$pred" | tee -a "$log"; fi
  done
done
echo "DONE gpu=$GPU $(date -Is)"
