#!/usr/bin/env python3
import subprocess
import sys
import time

print(f"[驱动] 测试 Traffic pred_len=720 内存优化版本")
print()

experiment = {
    'data': 'Traffic',
    'seq_len': 96,
    'pred_len': 720,
}

cmd = [
    sys.executable, 'run.py',
    '--task_name', 'long_term_forecast',
    '--is_training', '1',
    '--root_path', f'./dataset/{experiment["data"].lower()}/',
    '--data_path', f'{experiment["data"].lower()}.csv',
    '--model_id', f'{experiment["data"]}_{experiment["seq_len"]}_{experiment["pred_len"]}',
    '--model', 'RAFT',
    '--data', 'custom',
    '--features', 'M',
    '--seq_len', str(experiment["seq_len"]),
    '--label_len', '48',
    '--pred_len', str(experiment["pred_len"]),
    '--e_layers', '2',
    '--d_layers', '1',
    '--factor', '3',
    '--enc_in', '862',
    '--dec_in', '862',
    '--c_out', '862',
    '--des', 'baseline',
    '--itr', '1',
    '--train_epochs', '10',
    '--patience', '3',
    '--learning_rate', '0.0001',
    '--num_workers', '0',
    '--batch_size', '16',  # Reduce from 32 to 16 for large dataset
    '--top_k', '10',  # Reduce from 20 to 10 for memory
]

start_time = time.time()
print(f"[开始] Traffic seq=96 pred=720  ({time.strftime('%H:%M:%S')})")

result = subprocess.run(cmd, capture_output=False)

elapsed = (time.time() - start_time) / 60
if result.returncode == 0:
    print(f"[完成] Traffic pred=720 用时 {elapsed:.1f}min")
    sys.exit(0)
else:
    print(f"[失败] Traffic pred=720 returncode={result.returncode} 用时 {elapsed:.1f}min")
    sys.exit(1)
