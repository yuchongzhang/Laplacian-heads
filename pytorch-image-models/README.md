# Image classification

DeiT-B experiments for [Laplacian Heads Improve Transformers by Smoothing Token Representations](https://arxiv.org/abs/2602.09297).

- `timm/`: models, Laplacian attention, and training utilities; `configs/`: training configurations.
- `train.py`: ImageNet training; `train_cifar.py`: shared CIFAR training.
- `analysis/`: measurements, plots, and accuracy aggregation.

Run commands from this directory after activating the environment in [SETUP.md](../SETUP.md).

## Train

`L` counts Laplacian heads per block. Replace `3L` with `1L`, `6L`, `9L`, `11L`, or `12L`. Baselines: `deit3_base_qk_norm_patch4_32` (CIFAR) and `deit3_base_qk_norm_patch16_224` (ImageNet).

```bash
python train_cifar10.py --config configs/cifar10.yaml --data-dir /path/to/cifar10 --model deit3_base_3L_patch4_32 --seed 42 --output outputs/cifar10
python train_cifar100.py --config configs/cifar100.yaml --data-dir /path/to/cifar100 --model deit3_base_3L_patch4_32 --seed 42 --output outputs/cifar100
torchrun --standalone --nproc_per_node=4 train.py --config configs/deit3_base_12L_qk_norm.yaml --data-dir /path/to/imagenet --model deit3_base_3L_patch16_224 --seed 2026 --output outputs/imagenet
```

ImageNet needs matching `train/CLASS/` and `val/CLASS/` directories. CIFAR training will download missing data.

Evaluate an ImageNet checkpoint:

```bash
python validate.py --data-dir /path/to/imagenet --model deit3_base_3L_patch16_224 --checkpoint /path/to/model_best.pth.tar
```

## Measure

Use a checkpoint matching the model and dataset. CIFAR data must already be downloaded; ImageNet needs all 1,000 class folders under `val/`.

```bash
python -m analysis.measure anova --dataset cifar100 --data-root /path/to/cifar100 --model deit3_base_3L_patch4_32 --checkpoint /path/to/best.pth --output outputs/anova.json
python -m analysis.measure cosine --dataset cifar100 --data-root /path/to/cifar100 --model deit3_base_3L_patch4_32 --checkpoint /path/to/best.pth --output outputs/cosine.json
python -m analysis.measure nc --dataset cifar100 --data-root /path/to/cifar100 --model deit3_base_3L_patch4_32 --checkpoint /path/to/best.pth --output outputs/nc.json
python -m analysis.measure pca --dataset cifar100 --data-root /path/to/cifar100 --model deit3_base_3L_patch4_32 --checkpoint /path/to/best.pth --output outputs/pca.json
python -m analysis.measure simplex --dataset cifar100 --data-root /path/to/cifar100 --model deit3_base_3L_patch4_32 --checkpoint /path/to/best.pth --output outputs/simplex.json
python -m analysis.measure snr --dataset imagenet --data-root /path/to/imagenet --model deit3_base_3L_patch16_224 --checkpoint /path/to/model_best.pth.tar --output outputs/snr.json
```

Dataset choices: `cifar10`, `cifar100`, `imagenet`; use a `patch16_224` model for ImageNet. Common options: `--batch-size 32`, `--workers 4`, `--device cpu`, and `--max-images N` for a smaller run. `--sample-indices outputs/measurement.json` reuses saved image indices; keep dataset and population settings identical when comparing models.

| Metric | Available datasets | Useful options |
| --- | --- | --- |
| `anova` | CIFAR-10/100, ImageNet | Common options above |
| `cosine` | CIFAR-10/100, ImageNet | `--eval-images 512`, `--seed 0` |
| `snr` | ImageNet | `--eval-images 32`, `--seed 0` |
| `nc` | CIFAR-10/100, ImageNet | `--classes 1 7 9`; `--seed 0` selects default ImageNet classes |
| `pca` | CIFAR-10/100, ImageNet | `--classes 1 7 9`, `--samples-per-class 50`, `--seed 1002`, `--layer norm_final` or `block_0`–`block_11` |
| `simplex` | CIFAR-10/100, ImageNet | `--classes 1 7 9` (exactly three), `--samples-per-class 50`, `--seed 2027` |

## Plot and aggregate

Plot any measurement JSON; multiple inputs must contain the same metric. Aggregate training logs to an accuracy CSV. CIFAR logs need all 300 epochs; ImageNet summaries need epochs 270–299 and an adjacent `args.yaml`.

```bash
python -m analysis.plot outputs/anova.json --output outputs/anova.pdf
python -m analysis.plot outputs/cosine_baseline.json outputs/cosine_12L.json --output outputs/cosine.pdf
python -m analysis.aggregate outputs/cifar100/RUN/training_log.json --dataset cifar100 --output outputs/cifar100_accuracy.csv
python -m analysis.aggregate outputs/imagenet/RUN/summary.csv --output outputs/imagenet_accuracy.csv
```
