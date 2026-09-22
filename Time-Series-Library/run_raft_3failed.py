#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAFT 补跑失败的3个实验
Electricity pred_len=720, Traffic pred_len=336, Traffic pred_len=720
"""
import csv
import glob
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL = 'RAFT'
CSV_PATH = os.path.join(ROOT, 'raft_results_8datasets.csv')
LOG_DIR = os.path.join(ROOT, 'logs_raft')
FIELDNAMES = ['model', 'dataset', 'seq_len', 'pred_len', 'mse', 'mae']

SEQ_LEN = 96
DES = 'baseline'

# 只跑失败的3个
EXPERIMENTS = [
    ('Electricity', './dataset/electricity/', 'electricity.csv', 321, 720),
    ('Traffic', './dataset/traffic/', 'traffic.csv', 862, 336),
    ('Traffic', './dataset/traffic/', 'traffic.csv', 862, 720),
]


def find_result_dir(model_id):
    pattern = os.path.join(
        ROOT, 'results',
        'long_term_forecast_{}_{}_*_*_{}_0'.format(model_id, MODEL, DES))
    hits = glob.glob(pattern)
    return hits[0] if len(hits) == 1 else None


def append_row(row):
    exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def cleanup(result_dir):
    freed = 0
    if result_dir is None:
        return 0.0
    for name in ('pred.npy', 'true.npy'):
        p = os.path.join(result_dir, name)
        if os.path.exists(p):
            freed += os.path.getsize(p)
            os.remove(p)
    tr = os.path.join(ROOT, 'test_results', os.path.basename(result_dir))
    if os.path.isdir(tr):
        for name in os.listdir(tr):
            p = os.path.join(tr, name)
            if os.path.isfile(p):
                freed += os.path.getsize(p)
                os.remove(p)
    return freed / 1e9


def main():
    os.chdir(ROOT)
    os.makedirs(LOG_DIR, exist_ok=True)

    import torch
    if not torch.cuda.is_available():
        sys.exit('[驱动] 中止：torch 看不到 CUDA。请用 GPU 环境启动。')
    print('[驱动] GPU: {} | torch {} (cuda {})'.format(
        torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda), flush=True)

    print('[驱动] 补跑失败的 {} 个实验'.format(len(EXPERIMENTS)), flush=True)

    t_all = time.time()
    failures = []

    for name, root_path, data_path, enc_in, pred_len in EXPERIMENTS:
        model_id = '{}_{}_{}'.format(name, SEQ_LEN, pred_len)
        log_path = os.path.join(LOG_DIR, '{}_pl{}.log'.format(name, pred_len))

        data_arg = 'custom'

        cmd = [
            sys.executable, '-u', 'run.py',
            '--task_name', 'long_term_forecast',
            '--is_training', '1',
            '--root_path', root_path,
            '--data_path', data_path,
            '--model_id', model_id,
            '--model', MODEL,
            '--data', data_arg,
            '--features', 'M',
            '--seq_len', str(SEQ_LEN),
            '--label_len', '48',
            '--pred_len', str(pred_len),
            '--e_layers', '2',
            '--d_layers', '1',
            '--factor', '3',
            '--enc_in', str(enc_in),
            '--dec_in', str(enc_in),
            '--c_out', str(enc_in),
            '--des', DES,
            '--itr', '1',
            '--n_period', '3',
            '--topm', '20',
        ]

        print('\n[开始] {} seq={} pred={}  ({})'.format(
            name, SEQ_LEN, pred_len, time.strftime('%H:%M:%S')), flush=True)
        t0 = time.time()
        with open(log_path, 'w') as lf:
            ret = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
        dt = time.time() - t0

        result_dir = find_result_dir(model_id)
        metrics_path = os.path.join(result_dir, 'metrics.npy') if result_dir else ''
        if ret != 0 or not result_dir or not os.path.exists(metrics_path):
            print('[失败] {} pred_len={} returncode={} (日志: {})'.format(
                name, pred_len, ret, log_path), flush=True)
            failures.append((name, pred_len, ret, log_path))
            cleanup(result_dir)
            continue

        mae, mse = np.load(metrics_path)[:2]
        append_row({
            'model': MODEL,
            'dataset': name,
            'seq_len': SEQ_LEN,
            'pred_len': pred_len,
            'mse': '{:.6f}'.format(float(mse)),
            'mae': '{:.6f}'.format(float(mae)),
        })
        freed = cleanup(result_dir)
        print('[完成] {} pred={}  mse={:.6f}  mae={:.6f}  用时 {:.1f}min  释放 {:.2f}GB'.format(
            name, pred_len, mse, mae, dt / 60, freed), flush=True)

    print('\n[驱动] 全部结束, 总用时 {:.1f} 分钟'.format((time.time() - t_all) / 60), flush=True)
    if failures:
        print('[驱动] 以下实验失败:', flush=True)
        for f in failures:
            print('   {} pred_len={} rc={} 日志={}'.format(*f), flush=True)
        sys.exit(1)
    print('[驱动] 结果已追加到 {}'.format(CSV_PATH), flush=True)


if __name__ == '__main__':
    main()
