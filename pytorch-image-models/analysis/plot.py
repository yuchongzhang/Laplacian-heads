"""Plot saved analysis.measure JSON files."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    records = [json.loads(path.read_text()) for path in args.inputs]
    metric = records[0]['metadata']['metric']
    if any(r['metadata']['metric'] != metric for r in records):
        parser.error('Plot inputs must contain the same measurement.')
    if metric in ['pca', 'simplex']:
        fig, axes = plt.subplots(1, len(records), figsize=(5 * len(records), 4), squeeze=False)
        for ax, record in zip(axes[0], records):
            x = np.asarray(record['values']['projection'])
            labels = np.asarray(record['values']['labels'])
            for c in np.unique(labels):
                points = x[labels == c].reshape(-1, 2)
                ax.scatter(points[:, 0], points[:, 1], s=3, alpha=.5, label=str(c))
            ax.set_title(f"{record['metadata']['laplacian_heads']} Laplacian heads")
            ax.set_xlabel('Component 1'); ax.set_ylabel('Component 2')
            ax.set_aspect('equal', adjustable='datalim')
    elif metric == 'nc':
        names = list(records[0]['values'])
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        for ax, name in zip(axes.flat, names):
            pairs = sorted((r['metadata']['laplacian_heads'], r['values'][name]) for r in records)
            ax.plot(*zip(*pairs), marker='o'); ax.set_title(name); ax.set_xlabel('Laplacian heads')
    else:
        fig, ax = plt.subplots(figsize=(7, 4))
        if metric == 'anova':
            records.sort(key=lambda r: r['metadata']['laplacian_heads'])
            bottom = np.zeros(len(records))
            for name in ['between_class', 'within_class', 'within_seq']:
                values = np.array([r['values'][name + '_pct'] for r in records])
                ax.bar(range(len(records)), values, bottom=bottom, label=name.replace('_', ' '))
                bottom += values
            ax.set_xticks(range(len(records)), [r['metadata']['laplacian_heads'] for r in records])
            ax.set_xlabel('Laplacian heads'); ax.set_ylabel('Fraction of total variance')
        else:
            for record in records:
                values = record['values']
                ax.plot(range(len(values)), list(values.values()), marker='o', label=f"{record['metadata']['laplacian_heads']}L")
            ax.set_xticks(range(len(values)), list(values), rotation=45, ha='right')
            ax.set_ylabel('Centered CosSim' if metric == 'cosine' else 'Centered SNR')
        ax.legend()
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)


if __name__ == '__main__':
    main()
