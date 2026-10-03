import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from main.ETTh import ETTh_main_few_shot_reasoning as etth

DATASETS = [x for x in os.environ.get("MEMCAST_DATASETS", "").split(",") if x]
PRED_LENS = [int(x) for x in os.environ.get("MEMCAST_PRED_LENS", "96,192,336,720").split(",")]
SEQ_LEN = 96
NUM_SAMPLES = int(os.environ.get("MEMCAST_NUM_SAMPLES", "30"))
GPU = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
LOW_MEMORY = os.environ.get("MEMCAST_LOW_MEMORY", "0") == "1"
MODEL_NAME = "MemCast"
CSV_PATH = Path(os.environ.get("MEMCAST_CSV", "results/memcast_results.csv"))
FAILURE_PATH = Path(os.environ.get("MEMCAST_FAILURE_CSV", "results/memcast_failed_samples.csv"))
RESULTS_ROOT = Path(os.environ.get("MEMCAST_RESULTS_ROOT", "results/ETTh"))
DATA_DIR = Path("datasets")
API_KEY = os.environ.get("OPENAI_API_KEY", "")


def append_csv(path, row, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def metric_rows():
    if not CSV_PATH.exists():
        return set()
    with CSV_PATH.open(newline="") as f:
        return {(r["model"], r["dataset"], int(r["seq_len"]), int(r["pred_len"])) for r in csv.DictReader(f)}


def record_failure(dataset, pred_len, number, error):
    append_csv(FAILURE_PATH, {
        "model": MODEL_NAME, "dataset": dataset, "seq_len": SEQ_LEN,
        "pred_len": pred_len, "sample": number, "error": str(error)[:1000],
    }, ["model", "dataset", "seq_len", "pred_len", "sample", "error"])


def evaluate(dataset, pred_len, number):
    data = pd.read_csv(DATA_DIR / f"{dataset}.csv")
    truth_start = int(len(data) * 0.8) + number * 24 + SEQ_LEN
    truth = data.iloc[truth_start:truth_start + pred_len]["OT"].to_numpy(dtype=float)
    max_start = int(len(data) * 0.8) - SEQ_LEN - pred_len + 1
    saved_number = len(range(0, max_start, 24)) + number
    result_file = RESULTS_ROOT / "result_best" / dataset / (
        f"result_OT_{dataset}_{SEQ_LEN}_{pred_len}_{saved_number}.json"
    )
    if not result_file.exists():
        raise FileNotFoundError(result_file)
    with result_file.open() as f:
        payload = json.load(f)
    pred = np.asarray(payload[0].get("parsed_prediction", []), dtype=float)
    if len(pred) < pred_len:
        raise ValueError(f"prediction has {len(pred)}/{pred_len} values in {result_file}")
    return float(np.mean((truth - pred[:pred_len]) ** 2)), float(np.mean(np.abs(truth - pred[:pred_len])))


def run_sample(dataset, pred_len, number):
    lowmem_case = LOW_MEMORY and dataset in {"Electricity", "Traffic"} and pred_len == 720
    print(f"START dataset={dataset} seq_len={SEQ_LEN} pred_len={pred_len} sample={number} low_memory={lowmem_case}", flush=True)
    try:
        etth.ETTh_main_few_shot_reasoning(
            dataset, "OT", SEQ_LEN, pred_len, number, api_key=API_KEY,
            temperature=0.6, top_p=0.7, use_summary=False,
            num_similar_examples=1, num_few_shot_examples=1,
            _vision_analysis=False, _covariate_data=False,
        )
        mse, mae = evaluate(dataset, pred_len, number)
        print(f"DONE dataset={dataset} pred_len={pred_len} sample={number} mse={mse:.6f} mae={mae:.6f}", flush=True)
        return mse, mae
    except Exception as e:
        record_failure(dataset, pred_len, number, e)
        print(f"FAILED dataset={dataset} pred_len={pred_len} sample={number} error={e}", flush=True)
        return None


def main():
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")
    if not DATASETS:
        raise RuntimeError("MEMCAST_DATASETS is empty")
    done = metric_rows()
    print(f"runner pid={os.getpid()} gpu={GPU} model={MODEL_NAME} datasets={DATASETS} pred_lens={PRED_LENS}", flush=True)
    for dataset in DATASETS:
        for pred_len in PRED_LENS:
            key = (MODEL_NAME, dataset, SEQ_LEN, pred_len)
            if key in done:
                print(f"SKIP completed dataset={dataset} pred_len={pred_len}", flush=True)
                continue
            values = []
            for number in range(NUM_SAMPLES):
                metric = run_sample(dataset, pred_len, number)
                if metric is not None:
                    values.append(metric)
            if values:
                avg_mse = float(np.mean([x[0] for x in values]))
                avg_mae = float(np.mean([x[1] for x in values]))
                append_csv(CSV_PATH, {
                    "model": MODEL_NAME, "dataset": dataset, "seq_len": SEQ_LEN,
                    "pred_len": pred_len, "mse": f"{avg_mse:.6f}", "mae": f"{avg_mae:.6f}",
                }, ["model", "dataset", "seq_len", "pred_len", "mse", "mae"])
                print(f"SUMMARY dataset={dataset} pred_len={pred_len} successful={len(values)}/{NUM_SAMPLES} mse={avg_mse:.6f} mae={avg_mae:.6f}", flush=True)
            else:
                print(f"SUMMARY FAILED dataset={dataset} pred_len={pred_len} successful=0/{NUM_SAMPLES}", flush=True)


if __name__ == "__main__":
    main()

