"""Per-image uncentered spectra and component counts after final LayerNorm."""

import argparse
import csv
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dinov3.data.transforms import make_classification_eval_transform

from .common import image_paths, load_teacher, model_specs, sha256, write_csv, write_json


def eigenvalues(tokens):
    """All d eigenvalues of X.T @ X, descending; no token mean is subtracted."""
    if tokens.ndim != 3 or min(tokens.shape) < 1 or not torch.isfinite(tokens).all():
        raise ValueError("Expected finite, nonempty (images, patches, features) tokens")
    x = tokens.detach().double()
    options = {"driver": "gesvd"} if x.is_cuda else {}
    values = torch.linalg.svdvals(x, **options).square()
    return torch.nn.functional.pad(values, (0, x.shape[2] - values.shape[1]))


def component_counts(values, alpha):
    """Smallest per-image k capturing alpha of squared singular-value energy."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    cumulative = values.double().cumsum(-1)
    total = cumulative[:, -1]
    counts = (cumulative < alpha * total[:, None]).sum(-1) + 1
    return torch.where(total > 0, counts, 0)


class Images(Dataset):
    def __init__(self, root, rows):
        self.root = Path(root)
        self.rows = rows
        if not self.rows:
            raise ValueError("The sample manifest is empty")
        self.transform = make_classification_eval_transform(resize_size=256, crop_size=224)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.root / row["path"]
        if row.get("sha256") and sha256(path) != row["sha256"]:
            raise ValueError(f"Image hash mismatch: {path}")
        with Image.open(path) as image:
            return self.transform(image.convert("RGB"))


@torch.inference_mode()
def measure(model, loader, device, precision):
    parts = []
    for pixels in loader:
        context = torch.autocast(device.type, dtype=torch.bfloat16) if precision == "bf16" else nullcontext()
        with context:
            tokens = model.forward_features(pixels.to(device))["x_norm_patchtokens"]
        parts.append(eigenvalues(tokens).cpu())
    return torch.cat(parts), int(tokens.shape[1])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True, help="JSON model manifest")
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, help="Root for image paths in --samples")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--images", type=Path, nargs="+", help="Image files or folders (searched recursively)")
    selection.add_argument("--samples", type=Path, help="CSV with path and optional sha256 columns")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Measure only the first N selected images")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    device = torch.device(args.device)
    if args.images:
        rows = [{"path": str(path), "sha256": sha256(path)} for path in image_paths(args.images)[:args.limit]]
    else:
        if args.dataset_root is None:
            parser.error("--samples requires --dataset-root")
        with args.samples.open() as stream:
            rows = list(csv.DictReader(stream))
    if args.limit is not None:
        rows = rows[:args.limit]
    dataset = Images(args.dataset_root or Path(), rows)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                        shuffle=False, pin_memory=device.type == "cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "samples.csv", dataset.rows)
    spectra_rows, counts_rows, models = [], [], []
    for spec in model_specs(args.models):
        model, info = load_teacher(spec, args.checkpoint_root, device)
        values, tokens = measure(model, loader, device, args.precision)
        heads = int(spec["heads"])
        counts = {str(alpha): component_counts(values, alpha).numpy() for alpha in (.99, .999)}
        np.savez_compressed(args.output_dir / f"{heads}L_spectra.npz", eigenvalues=values.numpy(),
                            tokens_per_image=tokens, **{f"k_{alpha}": v for alpha, v in counts.items()})
        for rank, (mean, std) in enumerate(zip(values.mean(0), values.std(0, correction=0)), start=1):
            spectra_rows.append({"heads": heads, "rank": rank, "mean": float(mean), "std": float(std),
                                 "n_images": len(values), "tokens_per_image": tokens})
        for alpha, count in counts.items():
            counts_rows.append({"heads": heads, "energy_fraction": float(alpha), "n_images": len(count),
                                "mean_components": float(count.mean()), "std_components": float(count.std(ddof=0))})
        models.append(info)
        print(f"{heads}L: {len(values)} images, {tokens} patches, {values.shape[1]} eigenvalues", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_csv(args.output_dir / "spectrum.csv", spectra_rows)
    write_csv(args.output_dir / "component_counts.csv", counts_rows)
    write_json(args.output_dir / "metadata.json", {
        "measurement": "spectrum", "models": models, "n_images": len(dataset),
        "samples_sha256": sha256(args.output_dir / "samples.csv"),
        "representation": "final LayerNorm patch tokens; CLS/registers excluded", "centering": False,
        "inference_precision": args.precision, "spectral_precision": "float64",
        "standard_deviation": "population, ddof=0", "torch": torch.__version__, "device": str(device),
    })


if __name__ == "__main__":
    main()
