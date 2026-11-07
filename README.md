# Laplacian-Heads

Code for **[Laplacian Heads Improve Transformers by Smoothing Token Representations](https://arxiv.org/abs/2602.09297)**, by Yuchong Zhang and Vardan Papyan, NeurIPS 2026.

For query, key, and value matrices $Q$, $K$, and $V$, the row-wise attention matrix is

$$
P = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right),
$$

where $d_k$ is the query/key dimension and the model's attention mask is applied when needed.
A **Laplacian head** uses $I-P$ and outputs

$$
(I-P)V = V-PV,
$$

where $I$ is the identity matrix.

The paper contains experiments on three different tasks.

| Task | Code | Based on |
| --- | --- | --- |
| Image classification | [pytorch-image-models](pytorch-image-models/README.md) | [timm](https://github.com/huggingface/pytorch-image-models) |
| Language modeling | [nanochat](nanochat/README.md) | [nanochat](https://github.com/karpathy/nanochat) |
| Vision self-supervision | [dinov3](dinov3/README.md) | [DINOv3](https://github.com/facebookresearch/dinov3) |

Each task contains models, training, and measurements. See its README for file descriptions
and options, and [LICENSE](LICENSE) for licensing.

## Installation

```bash
bash setup/bootstrap.sh
source .venv/bin/activate
```

[SETUP.md](SETUP.md) describes the environment. Run each section below from the
named task directory and replace `/path/to/...` with your local paths.

## Image classification

Train with `L` Laplacian heads per block; use `0`, `3`, `6`, `9`, `11`, or `12`.

```bash
cd pytorch-image-models
CIFAR_DATA=/path/to/cifar
IMAGENET=/path/to/imagenet
L=6
if [ "$L" -eq 0 ]; then
    CIFAR_MODEL=deit3_base_qk_norm_patch4_32
    IMAGENET_MODEL=deit3_base_qk_norm_patch16_224
else
    CIFAR_MODEL="deit3_base_${L}L_patch4_32"
    IMAGENET_MODEL="deit3_base_${L}L_patch16_224"
fi

python train_cifar10.py --config configs/cifar10.yaml \
    --data-dir "$CIFAR_DATA" --model "$CIFAR_MODEL" --seed 42 --output outputs/cifar10
python train_cifar100.py --config configs/cifar100.yaml \
    --data-dir "$CIFAR_DATA" --model "$CIFAR_MODEL" --seed 42 --output outputs/cifar100
torchrun --standalone --nproc_per_node=4 train.py \
    --config configs/deit3_base_12L_qk_norm.yaml --data-dir "$IMAGENET" \
    --model "$IMAGENET_MODEL" --seed 2026 --output outputs/imagenet
```

Measure ANOVA, cosine similarity, pre-MLP SNR, neural collapse, PCA, and simplex projections:

```bash
CHECKPOINT=/path/to/model-checkpoint
for METRIC in anova cosine snr nc pca simplex; do
    python -m analysis.measure "$METRIC" --dataset imagenet \
        --data-root "$IMAGENET" --model "$IMAGENET_MODEL" \
        --checkpoint "$CHECKPOINT" --output "outputs/${METRIC}.json"
    python -m analysis.plot "outputs/${METRIC}.json" --output "outputs/${METRIC}.pdf"
done
python -m analysis.aggregate outputs/imagenet/*/summary.csv --output outputs/accuracy.csv
```

For CIFAR measurements, change `--dataset` to `cifar10` or `cifar100` and supply
the corresponding model, checkpoint, and data directory. SNR applies to ImageNet.
See the [classification README](pytorch-image-models/README.md) for metric options.

## Language modeling

Prepare data and a tokenizer, then choose a model depth and Laplacian head count:

```bash
cd nanochat
export NANOCHAT_BASE_DIR=/path/to/nanochat-data
python -m nanochat.dataset -n 380 -w 8
python -m scripts.tok_train --vocab-size=32768

DEPTH=30
L=6
SEED=1
TAG="d${DEPTH}-lap${L}-vanilla-s${SEED}"
case "$DEPTH" in
    12) BATCH=32; EVAL_EVERY=250; SAVE_EVERY=-1 ;;
    20) BATCH=16; EVAL_EVERY=250; SAVE_EVERY=-1 ;;
    30) BATCH=16; EVAL_EVERY=800; SAVE_EVERY=2000 ;;
esac
torchrun --standalone --nproc_per_node=4 -m scripts.base_train \
    --depth="$DEPTH" --laplacian-heads="$L" --seed="$SEED" --model-tag="$TAG" \
    --fp8 --window-pattern=SSSL \
    --target-param-data-ratio=12 --device-batch-size="$BATCH" \
    --eval-tokens=10485760 --eval-every="$EVAL_EVERY" --save-every="$SAVE_EVERY"
torchrun --standalone --nproc_per_node=4 -m scripts.base_eval \
    --model-tag="$TAG" --eval=core,bpb --device-batch-size=16 \
    --split-tokens=20971520 --max-per-task=-1 --seed="$SEED"
```

Head counts: d12 `{0,1,3,5,6}`, d20 `{0,3,5,7,10}`, d30 `{0,3,6,9,12,15}`.
Use the matching tokenizer when loading a checkpoint; see [SETUP.md](SETUP.md) for hardware requirements.

Measure the RMS-normalized residual stream after attention in the second-to-last block:

```bash
SHARD="$NANOCHAT_BASE_DIR/base_data_climbmix/shard_06542.parquet"
python -m scripts.prepare_measurement_data --input "$SHARD" --output outputs/contexts.npy
python -m scripts.measure_representations \
    --checkpoint-dir "$NANOCHAT_BASE_DIR/base_checkpoints/$TAG" \
    --data outputs/contexts.npy --output "outputs/${TAG}.json"
python -m scripts.plot_measurements "outputs/${TAG}.json" --output-dir outputs/figures
python -m scripts.plot_performance \
    "$NANOCHAT_BASE_DIR"/base_eval/*/base_model_*.json --output-dir outputs/performance
```

These commands produce spectra, component counts, next-token ANOVA, and BPB/CORE summaries.
Add `--center` to `scripts.measure_representations` for spectra centered over tokens
within each sequence. The default is uncentered; ANOVA is unaffected.

## DINO

Train with `0`, `1`, `3`, `5`, or `6` Laplacian heads per block:

```bash
cd dinov3
IMAGENET_PARQUET=/path/to/imagenet-parquet
TRAIN_DATASET="ImageNetParquet:split=TRAIN:root=$IMAGENET_PARQUET"
L=3
SEED=0
ARCH=vit_small_mixed
if [ "$L" -eq 0 ]; then ARCH=vit_small; fi
RUN="$PWD/outputs/vits_${L}L_seed${SEED}"
torchrun --standalone --nproc_per_node=4 -m dinov3.train.train \
    --config-file dinov3/configs/train/vits_imagenet.yaml \
    --output-dir "$RUN" --seed "$SEED" train.seed="$SEED" \
    train.dataset_path="$TRAIN_DATASET" student.arch="$ARCH" \
    student.num_vanilla_heads="$((6-L))"
```

Run ImageNet linear probing and ADE20K linear segmentation:

```bash
torchrun --standalone --nproc_per_node=1 -m dinov3.eval.linear \
    config_file=dinov3/configs/eval/linear_imagenet.yaml \
    model.config_file="$RUN/config.yaml" \
    model.pretrained_weights="$RUN/eval/training_142999/teacher_checkpoint.pth" \
    train.dataset="$TRAIN_DATASET" \
    train.val_dataset="ImageNetParquet:split=VAL:root=$IMAGENET_PARQUET" \
    output_dir="$RUN/linear"
ADE20K=/path/to/ADEChallengeData2016
python -m dinov3.data.prepare_ade20k --root "$ADE20K"
torchrun --standalone --nproc_per_node=1 -m dinov3.eval.segmentation.run \
    config=dinov3/configs/eval/segmentation_ade20k.yaml \
    model.config_file="$RUN/config.yaml" \
    model.pretrained_weights="$RUN/eval/training_142999/teacher_checkpoint.pth" \
    datasets.root="$ADE20K" output_dir="$RUN/segmentation"
EVALUATION_RUNS=/path/to/runs.json
python -m measurements.evaluation --runs "$EVALUATION_RUNS" \
    --results-root outputs --output-dir outputs/evaluation-summary
```

Measure patch spectra, component counts, and PCA maps:

```bash
CHECKPOINT_ROOT=/path/to/dino-checkpoints
MODELS=/path/to/models.json
IMAGES=/path/to/images
python -m measurements.spectrum --models "$MODELS" --checkpoint-root "$CHECKPOINT_ROOT" \
    --images "$IMAGES" --output-dir outputs/spectrum
python -m measurements.pca --models "$MODELS" --checkpoint-root "$CHECKPOINT_ROOT" \
    --images "$IMAGES" --output-dir outputs/pca
python -m measurements.plot --input-dir outputs/spectrum --output-dir outputs/spectrum-figures
python -m measurements.plot --input-dir outputs/pca --output-dir outputs/pca-figures
```

`--images` accepts image files or folders. See the [DINO README](dinov3/README.md)
for model/run manifest formats and image-selection options.
Add `--center` to `measurements.spectrum` to subtract each image's mean patch vector
before computing spectra and component counts; the default is uncentered.

## Citation

```bibtex
@misc{zhang2026laplacianheads,
  title={Laplacian Heads Improve Transformers by Smoothing Token Representations},
  author={Yuchong Zhang and Vardan Papyan},
  year={2026},
  eprint={2602.09297},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2602.09297}
}
```
