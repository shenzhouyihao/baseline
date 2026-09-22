#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PatchTST 长期预测实验驱动脚本
数据集: Weather / Electricity / Exchange / Traffic
窗口:   seq_len=96, pred_len in {96, 192, 336, 720}
输出:   patchtst_results.csv  (model,dataset,seq_len,pred_len,mse,mae)

超参数严格沿用 scripts/long_term_forecast/*_script/PatchTST.sh 官方配置。

特性:
  - 每个实验跑完立刻删除 results/<setting>/{pred,true}.npy，避免撑爆磁盘
  - 已完成的 (dataset, pred_len) 组合会跳过，支持中断后续跑
  - 逐条追加写入 CSV 并 flush，中途断电也不丢已完成结果

GPU 环境（重要）:
  /home/lzm/miniconda3/envs/tslib 的 torch 是 CPU-only 编译，Electricity 单个
  epoch 要 1.94 小时，全部 16 个实验约 7 天。本机有一块空闲 RTX 3090，改用
    /home/cxq/anaconda_envs/py3.9-torch2.5.1/bin/python   (torch 2.5.1+cu121)
  后 Electricity 单 epoch 降到 3.7 分钟（实测 6.15 -> 0.196 s/iter，31x），
  全部 16 个实验约 8 小时。峰值显存 5.7GB。启动方式：
    /home/cxq/anaconda_envs/py3.9-torch2.5.1/bin/python -u run_patchtst_4datasets.py
  子进程通过 sys.executable 继承同一解释器，所以不要用 tslib 的 python 启动。
"""
import csv
import glob
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL = 'PatchTST'
CSV_PATH = os.path.join(ROOT, 'patchtst_results.csv')
LOG_DIR = os.path.join(ROOT, 'logs_patchtst')
FIELDNAMES = ['model', 'dataset', 'seq_len', 'pred_len', 'mse', 'mae']

SEQ_LEN = 96
PRED_LENS = [96, 192, 336, 720]
DES = 'baseline'  # 与之前 ETT 实验保持一致

# (CSV里的数据集名, root_path, data_path, 变量数, 该数据集所有窗口共用的额外参数)
DATASETS = [
    ('Weather',     './dataset/weather/',       'weather.csv',        21, []),
    ('Electricity', './dataset/electricity/',   'electricity.csv',   321, ['--batch_size', '16']),
    ('Exchange',    './dataset/exchange_rate/', 'exchange_rate.csv',   8, []),
    ('Traffic',     './dataset/traffic/',       'traffic.csv',       862,
     ['--d_model', '512', '--d_ff', '512', '--top_k', '5', '--batch_size', '4']),
]

# 官方脚本中按 (dataset, pred_len) 单独指定的参数
PER_RUN_ARGS = {
    ('Weather', 96):  ['--n_heads', '4',  '--train_epochs', '3'],
    ('Weather', 192): ['--n_heads', '16', '--train_epochs', '3'],
    ('Weather', 336): ['--n_heads', '4',  '--batch_size', '128', '--train_epochs', '3'],
    ('Weather', 720): ['--n_heads', '4',  '--batch_size', '128', '--train_epochs', '3'],
    ('Exchange', 336): ['--train_epochs', '1'],
}


def find_result_dir(model_id):
    """d_model/n_heads 等会影响 setting 字符串，用 glob 定位本次实验的结果目录"""
    pattern = os.path.join(
        ROOT, 'results',
        'long_term_forecast_{}_{}_custom_*_{}_0'.format(model_id, MODEL, DES))
    hits = glob.glob(pattern)
    return hits[0] if len(hits) == 1 else None


def load_done():
    done = {}
    if not os.path.exists(CSV_PATH):
        return done
    with open(CSV_PATH, 'r', newline='') as f:
        for row in csv.DictReader(f):
            done[(row['dataset'], int(row['pred_len']))] = row
    return done


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
    """删除体积巨大的 pred/true 数组和可视化 pdf，保留 metrics.npy"""
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

    # 快速失败：用 CPU-only 的 torch 启动会让 16 个实验各自跑几十倍时长（或在
    # exp_basic._acquire_device() 之后的 .to('cuda') 处报错），这里提前拦住。
    import torch
    if not torch.cuda.is_available():
        sys.exit('[驱动] 中止：{} 的 torch 看不到 CUDA (torch {}, cuda build {})。\n'
                 '        请改用 /home/cxq/anaconda_envs/py3.9-torch2.5.1/bin/python 启动。'
                 .format(sys.executable, torch.__version__, torch.version.cuda))
    print('[驱动] GPU: {} | torch {} (cuda {})'.format(
        torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda), flush=True)

    done = load_done()

    total = len(DATASETS) * len(PRED_LENS)
    print('[驱动] {} 共 {} 个实验, 已完成 {}, 待运行 {}'.format(
        MODEL, total, len(done), total - len(done)), flush=True)

    t_all = time.time()
    failures = []

    for name, root_path, data_path, enc_in, ds_args in DATASETS:
        for pred_len in PRED_LENS:
            if (name, pred_len) in done:
                print('[跳过] {} pred_len={} 已有结果'.format(name, pred_len), flush=True)
                continue

            model_id = '{}_{}_{}'.format(name, SEQ_LEN, pred_len)
            log_path = os.path.join(LOG_DIR, '{}_pl{}.log'.format(name, pred_len))

            cmd = [
                sys.executable, '-u', 'run.py',
                '--task_name', 'long_term_forecast',
                '--is_training', '1',
                '--root_path', root_path,
                '--data_path', data_path,
                '--model_id', model_id,
                '--model', MODEL,
                '--data', 'custom',
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
                # 走 GPU：run.py 默认 use_gpu=True，不传 --no_use_gpu 即可。
                # sys.executable 会把启动本脚本的解释器传给子进程，所以必须用
                # 带 CUDA 的 torch 启动本脚本（见文件头 GPU 环境说明）。
            ] + ds_args + PER_RUN_ARGS.get((name, pred_len), [])

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
    print('[驱动] 结果已写入 {}'.format(CSV_PATH), flush=True)


if __name__ == '__main__':
    main()
