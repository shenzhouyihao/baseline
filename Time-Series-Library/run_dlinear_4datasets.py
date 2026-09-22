#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DLinear 长期预测实验驱动脚本
数据集: Weather / Electricity / Exchange / Traffic
窗口:   seq_len=96, pred_len in {96, 192, 336, 720}
输出:   dlinear_results.csv  (model,dataset,seq_len,pred_len,mse,mae)

特性:
  - 每个实验跑完立刻删除 results/<setting>/{pred,true}.npy，避免撑爆磁盘
  - 已完成的 (dataset, pred_len) 组合会跳过，支持中断后续跑
  - 逐条追加写入 CSV 并 flush，中途断电也不丢已完成结果

GPU 环境:
  使用 /home/cxq/anaconda_envs/py3.9-torch2.5.1/bin/python (torch 2.5.1+cu121)
  启动方式：/home/cxq/anaconda_envs/py3.9-torch2.5.1/bin/python -u run_dlinear_4datasets.py
"""
import csv
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(ROOT, 'dlinear_results.csv')
LOG_DIR = os.path.join(ROOT, 'logs_dlinear')
FIELDNAMES = ['model', 'dataset', 'seq_len', 'pred_len', 'mse', 'mae']

MODEL = 'DLinear'
SEQ_LEN = 96
PRED_LENS = [96, 192, 336, 720]

# (CSV里的数据集名, root_path, data_path, 变量数)
DATASETS = [
    ('Weather',     './dataset/weather/',       'weather.csv',        21),
    ('Electricity', './dataset/electricity/',   'electricity.csv',   321),
    ('Exchange',    './dataset/exchange_rate/', 'exchange_rate.csv',   8),
    ('Traffic',     './dataset/traffic/',       'traffic.csv',       862),
]

DES = 'baseline'  # 与之前 ETT 实验保持一致


def build_setting(model_id, enc_in, pred_len):
    """复刻 run.py 中的 setting 字符串，用于定位 results/<setting>/metrics.npy"""
    return (
        'long_term_forecast_{}_{}_{}_ftM_sl{}_ll48_pl{}_dm512_nh8_el2_dl1_'
        'df2048_expand2_dc4_fc3_ebtimeF_dtTrue_{}_0'
    ).format(model_id, MODEL, 'custom', SEQ_LEN, pred_len, DES)


def load_done():
    """读取已完成的实验，支持续跑"""
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


def cleanup(setting):
    """删除体积巨大的 pred/true 数组，保留 metrics.npy"""
    folder = os.path.join(ROOT, 'results', setting)
    freed = 0
    for name in ('pred.npy', 'true.npy'):
        p = os.path.join(folder, name)
        if os.path.exists(p):
            freed += os.path.getsize(p)
            os.remove(p)
    # 可视化 pdf 也一并清掉，数量多且本次实验用不到
    tr = os.path.join(ROOT, 'test_results', setting)
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

    # GPU 检查
    import torch
    if not torch.cuda.is_available():
        sys.exit('[驱动] 中止：{} 的 torch 看不到 CUDA (torch {}, cuda build {})。\n'
                 '        请改用 /home/cxq/anaconda_envs/py3.9-torch2.5.1/bin/python 启动。'
                 .format(sys.executable, torch.__version__, torch.version.cuda))
    print('[驱动] GPU: {} | torch {} (cuda {})'.format(
        torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda), flush=True)

    done = load_done()

    todo = [(d, p) for d in DATASETS for p in PRED_LENS
            if (d[0], p) not in done]
    print('[驱动] 共 {} 个实验, 已完成 {}, 待运行 {}'.format(
        len(DATASETS) * len(PRED_LENS), len(done), len(todo)), flush=True)

    t_all = time.time()
    failures = []

    for name, root_path, data_path, enc_in in DATASETS:
        for pred_len in PRED_LENS:
            if (name, pred_len) in done:
                print('[跳过] {} pred_len={} 已有结果'.format(name, pred_len), flush=True)
                continue

            model_id = '{}_{}_{}'.format(name, SEQ_LEN, pred_len)
            setting = build_setting(model_id, enc_in, pred_len)
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
                # GPU: run.py 默认 use_gpu=True
            ]

            print('\n[开始] {} seq={} pred={}  ({})'.format(
                name, SEQ_LEN, pred_len, time.strftime('%H:%M:%S')), flush=True)
            t0 = time.time()
            with open(log_path, 'w') as lf:
                ret = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
            dt = time.time() - t0

            metrics_path = os.path.join(ROOT, 'results', setting, 'metrics.npy')
            if ret != 0 or not os.path.exists(metrics_path):
                print('[失败] {} pred_len={} returncode={} (日志: {})'.format(
                    name, pred_len, ret, log_path), flush=True)
                failures.append((name, pred_len, ret, log_path))
                cleanup(setting)
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
            freed = cleanup(setting)
            print('[完成] {} pred={}  mse={:.6f}  mae={:.6f}  用时 {:.1f}s  释放 {:.2f}GB'.format(
                name, pred_len, mse, mae, dt, freed), flush=True)

    print('\n[驱动] 全部结束, 总用时 {:.1f} 分钟'.format((time.time() - t_all) / 60), flush=True)
    if failures:
        print('[驱动] 以下实验失败:', flush=True)
        for f in failures:
            print('   {} pred_len={} rc={} 日志={}'.format(*f), flush=True)
        sys.exit(1)
    print('[驱动] 结果已写入 {}'.format(CSV_PATH), flush=True)


if __name__ == '__main__':
    main()
