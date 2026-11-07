"""Load a frozen DINO teacher and portable experiment inputs."""

import csv
import hashlib
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from dinov3.configs.config import get_default_config
from dinov3.models import build_model_from_cfg

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def image_paths(inputs):
    """Expand files/folders in input order, sorting folders and removing duplicates."""
    paths = []
    for value in inputs:
        path = Path(value).expanduser().resolve()
        if path.is_dir():
            paths.extend(sorted(p.resolve() for p in path.rglob("*")
                                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS))
        elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(path)
        else:
            raise ValueError(f"Expected an image file or folder: {value}")
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise ValueError("No images found")
    return paths


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_specs(path):
    specs = read_json(path)["models"]
    heads = [int(spec["heads"]) for spec in specs]
    if not specs or len(heads) != len(set(heads)):
        raise ValueError("Provide one model for each distinct Laplacian head count")
    return sorted(specs, key=lambda spec: int(spec["heads"]))


def load_teacher(spec, checkpoint_root, device):
    """Load the backbone from a consolidated EMA teacher checkpoint, strictly."""
    root = Path(checkpoint_root)
    config_path = root / spec["config"]
    checkpoint_path = root / spec["checkpoint"]
    hashes = {"config_sha256": sha256(config_path), "checkpoint_sha256": sha256(checkpoint_path)}
    for key, actual in hashes.items():
        if spec.get(key) and spec[key] != actual:
            raise ValueError(f"{key} mismatch for {spec['run']}; the manifest does not match the supplied files")
    config = OmegaConf.merge(get_default_config(), OmegaConf.load(config_path))
    expected = 6 - int(config.student.num_vanilla_heads) if config.student.arch == "vit_small_mixed" else 0
    if config.student.arch not in ("vit_small", "vit_small_mixed") or expected != int(spec["heads"]):
        raise ValueError("The measurement manifest must match a ViT-S model's Laplacian head count")
    model, _ = build_model_from_cfg(config, only_teacher=True)
    model.to_empty(device=device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    state = {key.removeprefix("backbone."): value for key, value in checkpoint["teacher"].items()
             if key.startswith("backbone.")}
    model.load_state_dict(state, strict=True)
    model.eval()
    model.requires_grad_(False)
    info = {**spec, **hashes, "training_seed": int(config.train.seed)}
    return model, info
