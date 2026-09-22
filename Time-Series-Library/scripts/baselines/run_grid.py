#!/usr/bin/env python3
"""Run long-term forecasting baselines and maintain a resumable CSV table."""

import argparse
import csv
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


FIELDS = [
    "model",
    "dataset",
    "seq_len",
    "pred_len",
    "features",
    "mse",
    "mae",
    "seed",
    "run",
    "timestamp_utc",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--pred-lens", nargs="+", type=int, required=True)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--label-len", type=int, default=48)
    parser.add_argument("--features", default="M")
    parser.add_argument("--root-path", type=Path, default=Path("dataset/ETT-small"))
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("experiments/baseline_results.csv"))
    parser.add_argument("--logs-dir", type=Path, default=Path("experiments/logs"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no-use-gpu", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--d-layers", type=int, default=1)
    parser.add_argument("--factor", type=int, default=3)
    parser.add_argument("--down-sampling-layers", type=int)
    parser.add_argument("--down-sampling-method")
    parser.add_argument("--down-sampling-window", type=int)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-predictions", action="store_true")
    return parser.parse_args()


def load_rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def row_key(row):
    return (
        row["model"],
        row["dataset"],
        int(row["seq_len"]),
        int(row["pred_len"]),
        row["features"],
        int(row["seed"]),
        int(row["run"]),
    )


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def setting_name(args, dataset, pred_len):
    model_id = f"{dataset}_{args.seq_len}_{pred_len}"
    return (
        f"long_term_forecast_{model_id}_{args.model}_{dataset}_ft{args.features}"
        f"_sl{args.seq_len}_ll{args.label_len}_pl{pred_len}_dm{args.d_model}_nh{args.n_heads}"
        f"_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_expand2_dc4_fc{args.factor}"
        f"_ebtimeF_dtTrue_baseline_{args.run}"
    )


def run_one(args, project_root, dataset, pred_len):
    setting = setting_name(args, dataset, pred_len)
    log_path = args.logs_dir / args.model / f"{dataset}_{args.seq_len}_{pred_len}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        args.python,
        "-u",
        "run.py",
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--root_path", str(args.root_path.resolve()) + "/",
        "--data_path", f"{dataset}.csv",
        "--model_id", f"{dataset}_{args.seq_len}_{pred_len}",
        "--model", args.model,
        "--data", dataset,
        "--features", args.features,
        "--seq_len", str(args.seq_len),
        "--label_len", str(args.label_len),
        "--pred_len", str(pred_len),
        "--e_layers", str(args.e_layers),
        "--d_layers", str(args.d_layers),
        "--factor", str(args.factor),
        "--enc_in", "7",
        "--dec_in", "7",
        "--c_out", "7",
        "--d_model", str(args.d_model),
        "--d_ff", str(args.d_ff),
        "--des", "baseline",
        "--itr", "1",
        "--gpu", str(args.gpu),
        "--num_workers", str(args.num_workers),
        "--batch_size", str(args.batch_size),
        "--train_epochs", str(args.train_epochs),
        "--patience", str(args.patience),
        "--learning_rate", str(args.learning_rate),
    ]
    if args.no_use_gpu:
        command.append("--no_use_gpu")
    if args.down_sampling_layers is not None:
        command.extend(["--down_sampling_layers", str(args.down_sampling_layers)])
    if args.down_sampling_method is not None:
        command.extend(["--down_sampling_method", args.down_sampling_method])
    if args.down_sampling_window is not None:
        command.extend(["--down_sampling_window", str(args.down_sampling_window)])
    print(f"Running {args.model} {dataset} {args.seq_len}->{pred_len}", flush=True)
    with log_path.open("w") as log:
        log.write("COMMAND: " + " ".join(command) + "\n\n")
        result = subprocess.run(
            command,
            cwd=project_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"Experiment failed (see {log_path})")

    result_dir = project_root / "results" / setting
    metrics_path = result_dir / "metrics.npy"
    if not metrics_path.exists():
        raise RuntimeError(f"Missing metrics file: {metrics_path}")
    mae, mse = np.load(metrics_path)[:2]

    if not args.keep_predictions:
        for name in ("pred.npy", "true.npy"):
            (result_dir / name).unlink(missing_ok=True)
        shutil.rmtree(project_root / "test_results" / setting, ignore_errors=True)

    return float(mse), float(mae)


def main():
    args = parse_args()
    project_root = args.project_root.resolve() if args.project_root else Path(__file__).resolve().parents[2]
    os.chdir(project_root)
    args.output = args.output.resolve()
    args.logs_dir = args.logs_dir.resolve()
    args.root_path = args.root_path.resolve()

    rows = load_rows(args.output)
    completed = {row_key(row) for row in rows}
    for dataset in args.datasets:
        data_path = args.root_path / f"{dataset}.csv"
        if not data_path.exists():
            raise FileNotFoundError(data_path)
        for pred_len in args.pred_lens:
            key = (args.model, dataset, args.seq_len, pred_len, args.features, args.seed, args.run)
            if key in completed and not args.force:
                print(f"Skipping completed experiment: {dataset} {pred_len}")
                continue
            mse, mae = run_one(args, project_root, dataset, pred_len)
            rows = [row for row in rows if row_key(row) != key]
            rows.append({
                "model": args.model,
                "dataset": dataset,
                "seq_len": args.seq_len,
                "pred_len": pred_len,
                "features": args.features,
                "mse": f"{mse:.6f}",
                "mae": f"{mae:.6f}",
                "seed": args.seed,
                "run": args.run,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            rows.sort(key=lambda row: (row["model"], row["dataset"], int(row["pred_len"])))
            write_rows(args.output, rows)
            completed.add(key)
            print(f"Saved MSE={mse:.6f}, MAE={mae:.6f} to {args.output}", flush=True)


if __name__ == "__main__":
    main()
