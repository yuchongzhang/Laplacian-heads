"""Plot saved language measurements and export the plotted numbers as CSV."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(path.read_text()) for path in args.results]
    if any(record.get("schema_version") != 1 or "representation" not in record for record in records):
        parser.error("Expected measurement JSON files from scripts.measure_representations")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spectrum_rows, count_rows, anova_rows = [], [], []

    def save(fig, name):
        fig.tight_layout()
        for suffix in ("pdf", "png"):
            fig.savefig(args.output_dir / f"{name}.{suffix}", dpi=180)
        plt.close(fig)

    for depth in sorted({r["depth"] for r in records}):
        selected = [r for r in records if r["depth"] == depth]
        selected.sort(key=lambda r: (r["laplacian_heads"][0], r["model"]))
        for metric in ("spectrum", "anova"):
            relevant = [r for r in selected if metric in r]
            if not relevant:
                continue
            samples = {r[metric]["sample"]["sha256"] for r in relevant}
            representations = {json.dumps(r["representation"], sort_keys=True) for r in relevant}
            dtypes = {r["numerics"]["activation_dtype"] for r in relevant}
            if len(samples) != 1 or len(representations) != 1 or len(dtypes) != 1:
                parser.error(f"Cannot compare d{depth} {metric}: input samples, representation, or dtype differ")
            if metric == "spectrum":
                selections = {tuple(r[metric]["sample_indices"]) for r in relevant}
                if len(selections) != 1:
                    parser.error("Spectrum comparisons require identical sequence indices")
                fig, ax = plt.subplots(figsize=(6, 4))
                for record in relevant:
                    stats = record[metric]
                    mean, sd = np.asarray(stats["mean_eigenvalues"]), np.asarray(stats["std_eigenvalues"])
                    rank = np.arange(1, len(mean) + 1)
                    line, = ax.plot(rank, mean, label=record["model"])
                    lower = mean - sd
                    ax.fill_between(rank, lower, mean + sd, where=lower > 0, color=line.get_color(), alpha=.12)
                    spectrum_rows.extend({"model": record["model"], "depth": depth, "rank": int(i),
                                          "mean": float(mu), "population_sd": float(sigma)}
                                         for i, mu, sigma in zip(rank, mean, sd))
                ax.set(xscale="log", yscale="log", xlabel="Eigenvalue rank", ylabel="Eigenvalue of YᵀY")
                ax.legend(fontsize=7)
                save(fig, f"d{depth}_spectrum")
                for j, alpha in enumerate(relevant[0][metric]["alphas"]):
                    fig, ax = plt.subplots(figsize=(6, 4))
                    for record in relevant:
                        stats = record[metric]
                        if stats["alphas"] != relevant[0][metric]["alphas"]:
                            parser.error("Energy fractions differ between results")
                        heads = record["laplacian_heads"]
                        if len(set(heads)) != 1:
                            parser.error("Paper comparison plots expect a uniform head count across layers")
                        mean, sd = stats["mean_components"][j], stats["std_components"][j]
                        ax.errorbar(heads[0], mean, yerr=sd, fmt="o", capsize=3, label=record["model"])
                        count_rows.append({"model": record["model"], "depth": depth, "heads": heads[0],
                                           "fraction": alpha, "mean": mean, "population_sd": sd})
                    ax.set(xlabel="Laplacian heads per block", ylabel=f"Components explaining {100 * alpha:g}% energy")
                    save(fig, f"d{depth}_components_{100 * alpha:g}")
            else:
                fig, axes = plt.subplots(1, 2, figsize=(9, 4))
                for record in relevant:
                    stats = record[metric]
                    heads = record["laplacian_heads"]
                    if len(set(heads)) != 1:
                        parser.error("Paper comparison plots expect uniform head counts")
                    for ax, key in zip(axes, ("within_class", "between_class")):
                        fraction = stats[f"{key}_fraction"]
                        if fraction is not None:
                            ax.scatter(heads[0], 100 * fraction, label=record["model"])
                        ax.set(xlabel="Laplacian heads per block", ylabel=f"{key.replace('_', ' ').capitalize()} (% total)")
                    anova_rows.append({"model": record["model"], "depth": depth, "heads": heads[0],
                                       **{key: stats[key] for key in ("num_tokens", "num_classes", "within_class",
                                          "between_class", "total", "within_class_fraction", "between_class_fraction")}})
                save(fig, f"d{depth}_anova")
    write_csv(args.output_dir / "spectra.csv", spectrum_rows)
    write_csv(args.output_dir / "component_counts.csv", count_rows)
    write_csv(args.output_dir / "anova.csv", anova_rows)


if __name__ == "__main__":
    main()
