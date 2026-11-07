# DINO dataset preparation

Datasets are not bundled. Supply their paths explicitly.

## ImageNet training and linear probing

The paper recipe uses Hugging Face-format ImageNet-1k parquet files. Place `train-*.parquet` and `validation-*.parquet` in one directory. Files need `image` (struct with `bytes` and `path`) and integer `label` columns.

Use `ImageNetParquet:split=TRAIN:root=/path/to/data` for training and `split=VAL` for validation. First use creates `.cache_train/` or `.cache_val/` containing `images.bin`, `offsets.npy`, and `labels.npy`; allow disk space and write access. Later runs use the memory-mapped cache. Append `:extra=/path/to/cache` to choose a split-specific cache directory. In distributed training all ranks must see the same cache; rank zero creates it before others proceed.

`ImageNet:split=TRAIN:root=/path/to/images:extra=/path/to/metadata` accepts extracted images with the metadata arrays produced by `ImageNet.dump_extra()` in `dinov3/data/datasets/image_net.py`.

## Measurement images

Both measurement commands accept `--images photo.jpg /path/to/folder`. For explicit image lists, use `--samples` instead:

- Spectra: a CSV with a `path` column and optional `sha256` column. Supply `--dataset-root /path/to/images`; image paths are relative to that root.
- PCA: a JSON object with an `images` list. Each image needs a unique `name` and a `path`, relative to the JSON file's directory or absolute. Optional `sha256` values check image contents.

```json
{"images": [{"name": "photo", "path": "images/photo.jpg"}]}
```

PCA image entries may also specify `dataset: "imagenet"` or `dataset: "imagenet21k"`; their paths then use `--imagenet-root` or `--imagenet21k-root`. An image's optional `group` field supports `--group NAME`. A top-level `groups` list with entries such as `{"name": "comparison", "rows": ["photo"]}` controls plotting order.

## ADE20K

Extract `ADEChallengeData2016.zip` so the root contains `images/training`, `images/validation`, `annotations/training`, and `annotations/validation`. Generate the split lists required by the loader:

```bash
python -m dinov3.data.prepare_ade20k --root /path/to/ADEChallengeData2016
```

This checks 20,210 training and 2,000 validation images and their masks, then writes `ADE20K_object150_train.txt` and `ADE20K_object150_val.txt`. It rejects existing lists that disagree with the extracted images. Use this root as `datasets.root` for segmentation evaluation.
