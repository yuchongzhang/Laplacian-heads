"""Run one image-classification measurement on an explicitly chosen checkpoint.

Examples and measurement definitions are in the task README. Outputs contain the
actual head count, transforms, layer names, and selected dataset indices.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from .common import DATASETS, Features, dataset_for, loader_for, load_model, save_json
from .metrics import (cosine_similarity, signal_to_noise, pca_projection,
                      simplex_projection, nc_geometry, ncc_predictions)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('metric', choices=['anova', 'cosine', 'snr', 'nc', 'pca', 'simplex'])
    p.add_argument('--dataset', required=True, choices=DATASETS)
    p.add_argument('--data-root', required=True, type=Path)
    p.add_argument('--model', required=True)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, help='Sampling seed; defaults: PCA 1002, simplex 2027, otherwise 0')
    p.add_argument('--sample-indices', type=Path, help='JSON index list or a prior measurement JSON; overrides random image selection')
    p.add_argument('--max-images', type=int, help='Limit the measurement population, recorded in metadata (for quick checks)')
    p.add_argument('--eval-images', type=int, help='Sampled images after full-population centering; defaults: CosSim 512, SNR 32')
    p.add_argument('--classes', type=int, nargs='+', help='NC/PCA/simplex class IDs; original labels, in the desired order')
    p.add_argument('--samples-per-class', type=int, default=50, help='PCA/simplex balanced image count per class')
    p.add_argument('--layer', default='norm_final', help='PCA feature location: norm_final or block_0 ... block_11')
    args = p.parse_args(argv)
    if args.eval_images is None:
        args.eval_images = 512 if args.metric == 'cosine' else 32
    for key in ['batch_size', 'eval_images', 'samples_per_class']:
        if getattr(args, key) < 1:
            p.error(f'--{key.replace("_", "-")} must be positive')
    if args.max_images is not None and args.max_images < 1:
        p.error('--max-images must be positive')
    if args.workers < 0:
        p.error('--workers cannot be negative')
    if args.metric == 'snr' and args.dataset != 'imagenet':
        p.error('The paper reports pre-MLP SNR for ImageNet only.')
    if args.classes and args.metric not in ['nc', 'pca', 'simplex']:
        p.error('--classes applies to NC, PCA, and simplex measurements.')
    args.seed = args.seed if args.seed is not None else {'pca': 1002, 'simplex': 2027}.get(args.metric, 0)
    return args


def select_indices(args, dataset):
    labels = np.asarray(dataset.targets)
    available = list(range(DATASETS[args.dataset][0]))
    classes = args.classes
    if classes is None:
        if args.metric == 'pca':
            classes = list(range(10)) if args.dataset == 'cifar10' else ([0, 329, 658] if args.dataset == 'imagenet' else random.Random(args.seed).sample(available, 3))
        elif args.metric == 'simplex':
            classes = random.Random(args.seed).sample(available, 3)
        elif args.metric == 'nc':
            classes = sorted(random.Random(args.seed).sample(available, 200)) if args.dataset == 'imagenet' else available
    if classes is not None:
        if len(set(classes)) != len(classes) or any(c not in available for c in classes):
            raise ValueError('Class IDs must be unique and valid for this dataset.')
        if args.metric == 'simplex' and len(classes) != 3:
            raise ValueError('Simplex requires exactly three selected classes.')
    population = np.flatnonzero(np.isin(labels, classes)).tolist() if classes is not None else list(range(len(dataset)))
    if args.max_images:
        population = population[:args.max_images]
    if args.sample_indices:
        saved = json.loads(args.sample_indices.read_text())
        chosen = saved['metadata']['sample_indices'] if isinstance(saved, dict) else saved
        eligible = set(population)
        if not chosen or len(chosen) != len(set(chosen)) or any(i not in eligible for i in chosen):
            raise ValueError('Sample indices must be unique and belong to the selected population.')
    elif args.metric in ['pca', 'simplex']:
        rng = np.random.default_rng(args.seed)
        chosen = []
        for c in classes:
            pool = [i for i in population if labels[i] == c]
            if len(pool) < args.samples_per_class:
                raise ValueError(f'Class {c} has {len(pool)} eligible images, fewer than --samples-per-class.')
            chosen.extend(rng.choice(pool, size=args.samples_per_class, replace=False).tolist())
        rng.shuffle(chosen)
    elif args.metric in ['cosine', 'snr']:
        chosen = random.Random(args.seed).sample(population, min(args.eval_images, len(population)))
    else:
        chosen = population
    return population, chosen, classes


def feature_batches(model, loader, layers, device):
    with Features(model, layers) as capture, torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            yield capture.values, labels.to(device), logits


def class_statistics(model, loader, classes, device):
    """Accumulate token-weighted class means without retaining token tensors."""
    sums, counts = {}, {}
    for features, labels, _ in feature_batches(model, loader, ['norm_final'], device):
        x = features['norm_final']
        for c in labels.unique().tolist():
            tokens = x[labels == c].reshape(-1, x.shape[-1])
            sums[c] = sums.get(c, 0) + tokens.sum(0).double().cpu()
            counts[c] = counts.get(c, 0) + len(tokens)
    if classes is not None and set(classes) != set(sums):
        raise ValueError('The selected images must include every selected class; increase --max-images or supply valid sample indices.')
    means = {c: (sums[c] / counts[c]).float().to(device) for c in sums}
    total = sum(counts.values())
    global_mean = (sum(sums.values()) / total).float().to(device)
    return means, counts, global_mean


def anova(model, loader, device):
    means, counts, global_mean = class_statistics(model, loader, None, device)
    n_tokens = sum(counts.values())
    between = sum((means[c] - global_mean).norm().square().item() * counts[c] / n_tokens for c in means)
    total, within_seq, within_class, n_sequences = 0., 0., 0., 0
    for features, labels, _ in feature_batches(model, loader, ['norm_final'], device):
        x = features['norm_final']
        seq_mean = x.mean(1)
        total += (x - global_mean).norm(dim=-1).square().sum().item()
        within_seq += (x - seq_mean[:, None]).norm(dim=-1).square().sum().item()
        for c in labels.unique().tolist():
            within_class += (seq_mean[labels == c] - means[c]).norm(dim=-1).square().sum().item()
        n_sequences += len(x)
    values = dict(total_var=total / n_tokens, between_class_var=between,
                  within_class_var=within_class / n_sequences, within_seq_var=within_seq / n_tokens)
    if values['total_var'] <= 0:
        raise ValueError('ANOVA proportions are undefined for zero total variance.')
    for name in ['between_class', 'within_class', 'within_seq']:
        values[name + '_pct'] = values[name + '_var'] / values['total_var']
    values['anova_error'] = values['total_var'] - sum(values[name + '_var'] for name in ['between_class', 'within_class', 'within_seq'])
    return values


def centered_depth_measurement(model, population_loader, sample_loader, layers, metric, device):
    sums, counts = {name: 0 for name in layers}, {name: 0 for name in layers}
    for features, _, _ in feature_batches(model, population_loader, layers, device):
        for name, x in features.items():
            sums[name] = sums[name] + x.sum((0, 1)).double().cpu()
            counts[name] += x.shape[0] * x.shape[1]
    means = {name: (sums[name] / counts[name]).float().to(device) for name in layers}
    totals, n = {name: 0. for name in layers}, 0
    fn = cosine_similarity if metric == 'cosine' else signal_to_noise
    for features, labels, _ in feature_batches(model, sample_loader, layers, device):
        for name, x in features.items():
            totals[name] += fn(x - means[name]).double().sum().item()
        n += len(labels)
    return {name: totals[name] / n for name in layers}


def neural_collapse(model, loader, classes, device):
    means, _, _ = class_statistics(model, loader, classes, device)
    m = torch.stack([means[c] for c in classes])
    weights = model.get_classifier().weight.detach()[classes].float()
    result = nc_geometry(m, weights)
    matches, count = 0, 0
    for features, _, logits in feature_batches(model, loader, ['norm_final'], device):
        x = features['norm_final']
        prediction = ncc_predictions(x.reshape(-1, x.shape[-1]), m)
        # Historical NC4 compares every token's NCC against its image's CLS prediction.
        network = logits[:, classes].argmax(1).repeat_interleave(x.shape[1])
        matches += (prediction == network).sum().item()
        count += len(prediction)
    result['NC4_mismatch'] = 1 - matches / count
    return result


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    dataset, preprocessing = dataset_for(args.dataset, args.data_root, args.metric)
    population, indices, classes = select_indices(args, dataset)
    model = load_model(args.model, args.checkpoint, args.dataset, args.device)
    sample_loader = loader_for(dataset, indices, args.batch_size, args.workers)
    layers = ['norm_final']
    if args.metric == 'cosine':
        layers = [f'block_{i}' for i in range(len(model.blocks))] + ['norm_final']
    elif args.metric == 'snr':
        layers = [f'norm2_{i}' for i in range(len(model.blocks))]
    elif args.metric == 'pca':
        layers = [args.layer]
    if args.metric == 'anova':
        values = anova(model, sample_loader, args.device)
    elif args.metric in ['cosine', 'snr']:
        pop_loader = loader_for(dataset, population, args.batch_size, args.workers)
        values = centered_depth_measurement(model, pop_loader, sample_loader, layers, args.metric, args.device)
    elif args.metric == 'nc':
        values = neural_collapse(model, sample_loader, classes, args.device)
    else:
        tensors, labels = [], []
        for features, target, _ in feature_batches(model, sample_loader, layers, args.device):
            tensors.append(features[layers[0]].cpu())
            labels.extend(target.cpu().tolist())
        x = torch.cat(tensors)
        if args.metric == 'pca':
            projection = pca_projection(x)
        else:
            projection = simplex_projection(x, model.get_classifier().weight.detach().float().cpu()[classes])
        values = {'projection': projection.tolist(), 'labels': labels}
    first = model.blocks[0].attn
    metadata = {
        'metric': args.metric, 'dataset': args.dataset, 'model': args.model,
        'checkpoint': args.checkpoint.name, 'total_heads': first.num_heads,
        'laplacian_heads': first.num_heads - getattr(first, 'num_vanilla_heads', first.num_heads),
        'layers': layers, 'includes_cls_token': True, 'preprocessing': preprocessing,
        'sample_indices': indices, 'population_size': len(population),
        'population_limit': args.max_images, 'selected_classes': classes, 'sampling_seed': args.seed,
        'centering': {'anova': 'population_variance_decomposition', 'cosine': 'dataset_global_mean',
                      'snr': 'dataset_global_mean', 'nc': 'class_means_and_classifier_rows',
                      'pca': 'selected_token_global_mean', 'simplex': 'classifier_row_mean'}[args.metric],
        'protocol': 'image_measurements_v1',
    }
    save_json(args.output, {'metadata': metadata, 'values': values})
    print(f'Saved {args.metric} ({metadata["laplacian_heads"]} Laplacian heads) to {args.output}')


if __name__ == '__main__':
    main()
