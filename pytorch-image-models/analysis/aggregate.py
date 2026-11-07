"""Aggregate recorded per-seed accuracies without combining different recipes.

Accepts CIFAR training_log.json, ImageNet summary.csv (with args.yaml beside it),
or per-run accuracy CSVs with columns: column (recipe identifier), dataset,
lap_heads, seed, drop_path, weight_decay, top1, and metric.

CIFAR uses the best of epochs 0--299; ImageNet uses the best of epochs 270--299.
Duplicate runs of the same seed are averaged before calculating population
and sample standard deviations.
"""
import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
import re

import yaml


def effective_heads(model):
    match = re.search(r'_(\d+)L_', model)
    return int(match[1]) if match else 0


def training_row(path, dataset=None):
    if path.suffix == '.json':
        saved = json.loads(path.read_text())
        config, epochs = saved['config'], saved['epochs']
        dataset = config.get('dataset', dataset)
        if dataset not in ['cifar10', 'cifar100']:
            raise ValueError('For CIFAR logs, supply --dataset cifar10 or --dataset cifar100.')
        column, required = 'val_top1', set(range(300))
    else:
        config = yaml.safe_load(path.with_name('args.yaml').read_text())
        epochs = list(csv.DictReader(path.open()))
        dataset, column, required = 'imagenet', 'eval_top1', set(range(270, 300))
    by_epoch = {int(row['epoch']): float(row[column]) for row in epochs}
    if not required.issubset(by_epoch):
        raise ValueError(f'{path.name}: incomplete required epoch window; refusing a shorter selection window.')
    # Explicit hyperparameter identity avoids merging recipes that merely share wd/drop path.
    excluded = {'model', 'seed', 'data_dir', 'output', 'experiment', 'resume', 'config',
                'log_wandb', 'wandb_project', 'wandb_tags', 'workers', 'log_interval', 'ckpt_mode'}
    recipe = {k: v for k, v in config.items() if k not in excluded}
    recipe_id = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:12]
    return dict(column=dataset + '_' + recipe_id, dataset=dataset,
                lap_heads=effective_heads(config['model']), seed=config['seed'],
                drop_path=config['drop_path'], weight_decay=config['weight_decay'],
                top1=max(by_epoch[e] for e in required),
                metric='best over epochs 270-299' if dataset == 'imagenet' else 'best over 300 epochs')


def aggregate(rows):
    seed_values = defaultdict(list)
    for row in rows:
        key = (row['column'], row['dataset'], int(row['lap_heads']), float(row['drop_path']),
               float(row['weight_decay']), row['metric'])
        seed_values[key + (int(row['seed']),)].append(float(row['top1']))
    groups = defaultdict(list)
    for key, values in seed_values.items():
        groups[key[:-1]].append(statistics.mean(values))
    output = []
    for key, values in sorted(groups.items()):
        row = dict(zip(['cohort', 'dataset', 'lap_heads', 'drop_path', 'weight_decay', 'metric'], key))
        row.update(mean=statistics.mean(values), population_sd=statistics.pstdev(values),
                   sample_sd=statistics.stdev(values) if len(values) > 1 else None, n=len(values))
        output.append(row)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path, help='Training logs or per-run accuracy CSVs')
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100', 'imagenet'], help='Task for CIFAR logs without dataset metadata; unnecessary for per-run CSVs')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    rows = []
    for path in args.inputs:
        if path.suffix == '.csv':
            records = list(csv.DictReader(path.open()))
            if records and 'top1' in records[0] and 'column' in records[0]:
                rows.extend(records); continue
        rows.append(training_row(path, args.dataset))
    result = aggregate(rows)
    if not result:
        parser.error('No complete observations found.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0]))
        writer.writeheader(); writer.writerows(result)


if __name__ == '__main__':
    main()
