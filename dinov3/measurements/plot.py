"""Render the paper's DINO spectra, component counts, and PCA grids."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .common import read_json

COLORS = {0: "#4c78a8", 1: "#e45756", 3: "#72b7b2", 5: "#f58518", 6: "#54a24b"}


def label(heads):
    return "ViT-S" if heads == 0 else f"ViT-S-{heads}L"


def save(fig, directory, name):
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def spectra(source, output):
    with (source / "spectrum.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    heads = sorted({int(row["heads"]) for row in rows})
    selections = [("spectrum_all", heads)]
    if 0 in heads and 6 in heads:
        selections.append(("spectrum_baseline_vs_6L", [0, 6]))
    for name, selected in selections:
        fig, ax = plt.subplots(figsize=(6, 4))
        for head in selected:
            values = [row for row in rows if int(row["heads"]) == head and int(row["rank"]) <= int(row["tokens_per_image"])]
            rank, mean, std = np.array([[float(row[key]) for row in values] for key in ("rank", "mean", "std")])
            color = COLORS.get(head)
            ax.plot(rank, mean, label=label(head), color=color)
            # Only the display is clipped: numeric means/SDs stay unchanged.
            lower = np.maximum(mean - std, np.maximum(mean, np.finfo(float).tiny) * 1e-3)
            ax.fill_between(rank, lower, mean + std, alpha=.15, color=color)
        ax.set(xlabel="Eigenvalue rank", ylabel="Eigenvalue", yscale="log")
        ax.legend(frameon=False)
        save(fig, output, name)
    with (source / "component_counts.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    for alpha in sorted({float(row["energy_fraction"]) for row in rows}):
        selected = [row for row in rows if float(row["energy_fraction"]) == alpha]
        selected.sort(key=lambda row: int(row["heads"]))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([label(int(row["heads"])) for row in selected], [float(row["mean_components"]) for row in selected],
               yerr=[float(row["std_components"]) for row in selected], capsize=3,
               color=[COLORS.get(int(row["heads"]), "gray") for row in selected])
        ax.set(ylabel=f"Components for {100 * alpha:g}% energy")
        save(fig, output, f"component_counts_{100 * alpha:g}")


def pca_grids(source, output, metadata):
    heads = [int(model["heads"]) for model in metadata["models"]]
    for group in metadata["groups"]:
        rows = []
        for name in group["rows"]:
            directory = source / "maps" / name
            paths = [directory / "original.png"] + [directory / f"{head}L.png" for head in heads]
            rows.append([np.asarray(Image.open(path).convert("RGB")) for path in paths])
        columns = len(heads) + 1
        width, gap, row_gap, header = 396., 2.4, 3., 14.
        cell = (width - (columns - 1) * gap) / columns
        heights = [cell * row[0].shape[0] / row[0].shape[1] for row in rows]
        height = header + sum(heights) + row_gap * (len(rows) - 1)
        fig = plt.figure(figsize=(width / 72, height / 72))
        for column, title in enumerate(["Original"] + [label(head) for head in heads]):
            fig.text((column * (cell + gap) + cell / 2) / width, (height - 7.5) / height,
                     title, ha="center", va="center", fontsize=9)
        top = height - header
        for row, row_height in zip(rows, heights):
            top -= row_height
            for column, pixels in enumerate(row):
                ax = fig.add_axes((column * (cell + gap) / width, top / height, cell / width, row_height / height))
                ax.imshow(pixels, interpolation="none", resample=False)
                ax.axis("off")
            top -= row_gap
        save(fig, output, f"pca_{group['name']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = read_json(args.input_dir / "metadata.json")
    with plt.rc_context({"font.family": "serif", "pdf.fonttype": 42, "image.composite_image": False}):
        if metadata["measurement"] == "spectrum":
            spectra(args.input_dir, args.output_dir)
        elif metadata["measurement"] == "pca":
            pca_grids(args.input_dir, args.output_dir, metadata)
        else:
            parser.error("Expected spectrum or PCA measurement output")


if __name__ == "__main__":
    main()
