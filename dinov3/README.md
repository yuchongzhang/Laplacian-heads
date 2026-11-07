# DINO experiments

Code for [Laplacian Heads Improve Transformers by Smoothing Token Representations](https://arxiv.org/abs/2602.09297).

- `dinov3/layers/attention.py`: attention and Laplacian heads.
- `dinov3/models/`: ViT-S/16 backbones.
- `dinov3/train/`: self-supervised training.
- `dinov3/eval/`: frozen ImageNet classification and ADE20K segmentation.
- `dinov3/configs/`: training and evaluation configurations.
- `measurements/`: spectra, PCA maps, evaluation summaries, and plotting.
- [LICENSE.md](LICENSE.md): license for the [DINOv3](https://github.com/facebookresearch/dinov3) source.

Run commands from this directory after [setup](../SETUP.md). See [DATASETS.md](DATASETS.md) for dataset layouts.

## Training

```bash
IMAGENET_PARQUET=/path/to/imagenet-parquet
L=3  # 0, 1, 3, 5, or 6 Laplacian heads
SEED=0
RUN="$PWD/outputs/vits_${L}L_seed${SEED}"
ARCH=vit_small_mixed
if [ "$L" -eq 0 ]; then ARCH=vit_small; fi

torchrun --standalone --nproc_per_node=4 -m dinov3.train.train \
  --config-file dinov3/configs/train/vits_imagenet.yaml \
  --output-dir "$RUN" --seed "$SEED" \
  train.seed="$SEED" \
  train.dataset_path="ImageNetParquet:split=TRAIN:root=$IMAGENET_PARQUET" \
  student.arch="$ARCH" student.num_vanilla_heads="$((6-L))"
```

## Frozen evaluation

```bash
torchrun --standalone --nproc_per_node=1 -m dinov3.eval.linear \
  config_file=dinov3/configs/eval/linear_imagenet.yaml \
  model.config_file="$RUN/config.yaml" \
  model.pretrained_weights="$RUN/eval/training_142999/teacher_checkpoint.pth" \
  train.dataset="ImageNetParquet:split=TRAIN:root=$IMAGENET_PARQUET" \
  train.val_dataset="ImageNetParquet:split=VAL:root=$IMAGENET_PARQUET" \
  output_dir="$RUN/linear"

ADE20K=/path/to/ADEChallengeData2016
python -m dinov3.data.prepare_ade20k --root "$ADE20K"
torchrun --standalone --nproc_per_node=1 -m dinov3.eval.segmentation.run \
  config=dinov3/configs/eval/segmentation_ade20k.yaml \
  model.config_file="$RUN/config.yaml" \
  model.pretrained_weights="$RUN/eval/training_142999/teacher_checkpoint.pth" \
  datasets.root="$ADE20K" output_dir="$RUN/segmentation"
```

Evaluations write accuracy/mIoU results. To summarize runs across training seeds, create `runs.json`:

```json
{"runs": [{"model": "ViT-S-3L", "training_seed": 0, "linear": "vits_3L_seed0/linear", "segmentation": "vits_3L_seed0/segmentation"}]}
```

```bash
python -m measurements.evaluation --runs runs.json --results-root outputs \
  --output-dir outputs/evaluation_summary
```

Alternatively, use `--per-run results.csv` with columns `model,training_seed,task,metric,value`. The summary writes means and sample standard deviations.

## Spectra and PCA

Create `models.json` with one entry per model. Config and checkpoint paths are relative to `--checkpoint-root`; PCA requires a baseline with `heads: 0`.

```json
{"models": [
  {"run": "vits_0L_seed0", "heads": 0, "config": "vits_0L_seed0/config.yaml", "checkpoint": "vits_0L_seed0/eval/training_142999/teacher_checkpoint.pth", "label": "ViT-S"},
  {"run": "vits_3L_seed0", "heads": 3, "config": "vits_3L_seed0/config.yaml", "checkpoint": "vits_3L_seed0/eval/training_142999/teacher_checkpoint.pth", "label": "ViT-S-3L"}
]}
```

Both commands require `--models` and an image selection. `--images` accepts one or more files or folders; folders are searched recursively.

```bash
CHECKPOINTS=/path/to/checkpoints

python -m measurements.spectrum --models models.json \
  --checkpoint-root "$CHECKPOINTS" --images /path/to/images \
  --output-dir outputs/spectrum
python -m measurements.plot \
  --input-dir outputs/spectrum --output-dir outputs/spectrum/figures

python -m measurements.pca --models models.json \
  --checkpoint-root "$CHECKPOINTS" --images /path/to/images \
  --output-dir outputs/pca
python -m measurements.plot \
  --input-dir outputs/pca --output-dir outputs/pca/figures
```

Spectra produce per-image eigenvalues/component counts and CSV summaries; PCA produces scores and color maps. Plotting writes PNG/PDF figures. To select images with external CSV/JSON manifests, use `--samples` as described in [DATASETS.md](DATASETS.md#measurement-images).
