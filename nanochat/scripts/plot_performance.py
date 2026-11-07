"""Aggregate validation BPB and CORE across seeds and plot each model depth."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="*", help="JSON files from scripts.base_eval")
    parser.add_argument("--csv", type=Path, help="CSV with model, depth, heads, seed, val_bpb, core_score columns")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.csv) == bool(args.results):
        parser.error("Supply evaluation JSON files or --csv, but not both")
    rows = []
    if args.csv:
        with args.csv.open() as handle:
            for record in csv.DictReader(handle):
                rows.append({key: (record[key] if key == "model" else int(record[key]) if key in
                                  ("depth", "heads", "seed") else float(record[key]))
                             for key in ("model", "depth", "heads", "seed", "val_bpb", "core_score")})
    else:
        for path in args.results:
            record = json.loads(path.read_text())
            config = record["model_config"]
            for option in ("use_ve", "use_resid_lambdas", "use_x0", "use_smear", "use_backout"):
                if config.get(option, False):
                    parser.error(f"{path}: expected the paper's vanilla architecture")
            counts = config["laplacian_heads"]
            if len(set(counts)) != 1 or record["training_seed"] is None:
                parser.error(f"{path}: expected a uniform head count and recorded training seed")
            if record["val_bpb"] is None or record["core_score"] is None:
                parser.error(f"{path}: run base_eval --eval=core,bpb before plotting")
            if record["split_tokens"] != 20971520 or record["max_per_task"] != -1:
                parser.error(f"{path}: use the paper's 20971520 validation tokens and all CORE examples")
            rows.append({"model": record["model"], "depth": config["n_layer"], "heads": counts[0],
                         "seed": record["training_seed"], "val_bpb": record["val_bpb"],
                         "core_score": record["core_score"]})
    if not rows:
        parser.error("No performance measurements found")
    keys = [(r["depth"], r["heads"], r["seed"]) for r in rows]
    if len(keys) != len(set(keys)):
        parser.error("Multiple results for the same depth, head count and seed; choose one checkpoint per run")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["depth"], row["heads"]].append(row)
    summaries = []
    for (depth, heads), group in sorted(grouped.items()):
        row = {"depth": depth, "heads": heads, "seeds": ",".join(str(r["seed"]) for r in sorted(group, key=lambda r:r["seed"])), "n": len(group)}
        for metric in ("val_bpb", "core_score"):
            values = [r[metric] for r in group]
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_sample_sd"] = stdev(values) if len(values) > 1 else None
        summaries.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "performance_runs.csv", sorted(rows, key=lambda r:(r["depth"], r["heads"], r["seed"])))
    write_csv(args.output_dir / "performance_summary.csv", summaries)
    for depth in sorted({r["depth"] for r in summaries}):
        selected = [r for r in summaries if r["depth"] == depth]
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for ax, metric, label in zip(axes, ("val_bpb", "core_score"), ("Validation BPB", "CORE score")):
            ax.plot([r["heads"] for r in selected], [r[f"{metric}_mean"] for r in selected], "o-")
            for row in selected:
                sd = row[f"{metric}_sample_sd"]
                if sd is not None:
                    ax.errorbar(row["heads"], row[f"{metric}_mean"], yerr=sd, color="C0", capsize=3)
            ax.set(xlabel="Laplacian heads per block", ylabel=label, title=f"d{depth}")
        fig.tight_layout()
        for suffix in ("pdf", "png"):
            fig.savefig(args.output_dir / f"d{depth}_performance.{suffix}", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
