#!/usr/bin/env python3
"""Recompute normalized MemCast metrics from saved JSON predictions.

The script is read-only with respect to existing results. It writes no output
unless --output is supplied, in which case only the derived CSV is created.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = [
    "ETTh1", "ETTh2", "ETTm1", "ETTm2",
    "Weather", "Electricity", "Exchange", "Traffic",
]
PRED_LENS = [96, 192, 336, 720]


def expected_index_offset(data_len: int, pred_len: int) -> int:
    end_idx = int(data_len * 0.8)
    max_start = end_idx - 96 - pred_len + 1
    return len(range(0, max_start, 24))


def load_prediction(path: Path, pred_len: int):
    try:
        payload = json.loads(path.read_text())
        if not payload or not isinstance(payload[0], dict):
            return None
        values = np.asarray(payload[0].get("parsed_prediction", []), dtype=float)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    if values.size < pred_len or not np.isfinite(values[:pred_len]).all():
        return None
    return values[:pred_len]


def recompute(root: Path, planned_samples: int):
    result_roots = [root / "results/ETTh/opus46_direct720/result_best", root / "results/ETTh/result_best", root / "results/ETTh/terra_bounded/result_best"]
    rows = []
    for dataset in DATASETS:
        data = pd.read_csv(root / "datasets" / f"{dataset}.csv")
        target = data["OT"].to_numpy(dtype=float)
        train = target[: int(len(target) * 0.7)]
        train_mean = float(np.mean(train))
        train_std = float(np.std(train))
        if train_std == 0:
            raise ValueError(f"{dataset}: training OT std is zero")

        for pred_len in PRED_LENS:
            offset = expected_index_offset(len(target), pred_len)
            expected_paths = []
            for sample in range(planned_samples):
                filename = f"result_OT_{dataset}_96_{pred_len}_{offset + sample}.json"
                path = next((candidate / dataset / filename for candidate in result_roots if (candidate / dataset / filename).exists()), result_roots[0] / dataset / filename)
                expected_paths.append(path)

            errors = []
            valid_files = 0
            invalid_files = 0
            for sample, path in enumerate(expected_paths):
                if not path.exists():
                    continue
                pred = load_prediction(path, pred_len)
                if pred is None:
                    invalid_files += 1
                    continue
                start = int(len(target) * 0.8) + sample * 24 + 96
                truth = target[start : start + pred_len]
                if truth.size != pred_len or not np.isfinite(truth).all():
                    invalid_files += 1
                    continue
                errors.append((pred - truth) / train_std)
                valid_files += 1

            if errors:
                normalized = np.concatenate(errors)
                mse_norm = float(np.mean(normalized ** 2))
                mae_norm = float(np.mean(np.abs(normalized)))
            else:
                mse_norm = float("nan")
                mae_norm = float("nan")

            theoretical_windows = max(0, len(range(0, int(len(target) * 0.2) - pred_len + 1, 24)))
            rows.append({
                "dataset": dataset,
                "pred_len": pred_len,
                "train_mean": train_mean,
                "train_std": train_std,
                "successful": valid_files,
                "planned": planned_samples,
                "missing_or_invalid": planned_samples - valid_files,
                "available_test_windows": theoretical_windows,
                "mse_norm": mse_norm,
                "mae_norm": mae_norm,
                "invalid_existing_files": invalid_files,
            })
    return rows


def print_table(rows):
    fields = [
        "dataset", "pred_len", "successful", "planned",
        "available_test_windows", "train_mean", "train_std", "mse_norm", "mae_norm",
    ]
    print(",".join(fields))
    for row in rows:
        values = []
        for field in fields:
            value = row[field]
            if isinstance(value, float):
                values.append("NA" if not np.isfinite(value) else f"{value:.6f}")
            else:
                values.append(str(value))
        print(",".join(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--planned-samples", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = recompute(args.root, args.planned_samples)
    print_table(rows)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()



