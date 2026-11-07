"""Summarize ImageNet linear accuracy and ADE20K mIoU across training seeds."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from .common import read_json, write_csv


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        if (row["task"], row["metric"]) in (("linear", "accuracy"), ("segmentation", "mIoU")):
            groups[(row["model"], row["task"], row["metric"])].append(row)
    result = []
    for (model, task, metric), group in sorted(groups.items()):
        seeds = [int(row["training_seed"]) for row in group]
        values = [float(row["value"]) for row in group]
        if len(seeds) != len(set(seeds)) or any(not math.isfinite(v) or not 0 <= v <= 100 for v in values):
            raise ValueError("Expected one finite percentage per model/task/training seed")
        result.append({"model": model, "task": task, "metric": metric, "n": len(values),
                       "training_seeds": json.dumps(sorted(seeds)), "mean": statistics.mean(values),
                       "sample_std": statistics.stdev(values) if len(values) > 1 else "", "ddof": 1})
    if not result:
        raise ValueError("No linear accuracy or segmentation mIoU rows found")
    return result


def collect(runs, root):
    rows = []
    for run in read_json(runs)["runs"]:
        for task, filename in (("linear", "results-linear.csv"), ("segmentation", "results-semantic-segmentation.csv")):
            path = root / run[task] / filename
            with path.open() as stream:
                records = list(csv.DictReader(stream))
            if len(records) != 1:
                raise ValueError(f"Expected one completed result row: {path}")
            record = records[0]
            key = "mIoU"
            if task == "linear":
                keys = [key for key in record if key.endswith("_accuracy")]
                if len(keys) != 1 or not record.get("best_classifier"):
                    raise ValueError(f"Missing selected classifier or accuracy: {path}")
                key = keys[0]
            rows.append({"model": run["model"], "training_seed": run["training_seed"], "task": task,
                         "metric": "accuracy" if task == "linear" else "mIoU", "value": float(record[key])})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--per-run", type=Path, help="A CSV with model,training_seed,task,metric,value")
    source.add_argument("--runs", type=Path, help="JSON run manifest with relative linear and segmentation directories")
    parser.add_argument("--results-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.runs:
        rows = collect(args.runs, args.results_root)
    else:
        with args.per_run.open() as stream:
            rows = list(csv.DictReader(stream))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_run.csv", rows)
    write_csv(args.output_dir / "summary.csv", summarize(rows))


if __name__ == "__main__":
    main()
