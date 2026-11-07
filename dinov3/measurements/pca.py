"""Centered, whitened PCA maps aligned to the baseline for each image."""

import argparse
import itertools
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from torchvision.transforms import functional as TF

from .common import image_paths, load_teacher, model_specs, read_json, sha256, write_json


def resize_image(image, height=768, patch_size=16):
    width, original_height = image.size
    grid = (height // patch_size, int(width * height / (original_height * patch_size)))
    if min(grid) < 1:
        raise ValueError("Image is too small for the patch grid")
    pixels = TF.to_tensor(TF.resize(image, (grid[0] * patch_size, grid[1] * patch_size)))
    return pixels, grid


def project(tokens):
    """Fit the paper's three centered PCs with unit sample variance."""
    x = np.asarray(tokens, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 3 or not np.isfinite(x).all():
        raise ValueError("PCA requires at least three finite patches and features")
    # Use a fixed seed for sklearn's automatic solver.
    np.random.seed(0)
    pca = PCA(n_components=3, whiten=True).fit(x)
    if (pca.explained_variance_ <= 0).any():
        raise ValueError("The image has fewer than three nonzero principal directions")
    scores = pca.transform(x)
    return scores, pca


def align(reference, scores):
    """Choose only signs/order, maximizing mean spatial Pearson correlation."""
    reference = reference - reference.mean(0)
    centered = scores - scores.mean(0)
    denominator = np.linalg.norm(reference, axis=0)[:, None] * np.linalg.norm(centered, axis=0)[None, :]
    if (denominator == 0).any() or reference.shape != scores.shape:
        raise ValueError("Alignment needs three varying score maps on matching grids")
    correlation = (reference.T @ centered) / denominator
    variants = [(p, s) for p in itertools.permutations(range(3))
                for s in itertools.product((1, -1), repeat=3)]
    objectives = [np.mean(correlation[np.arange(3), p] * s) for p, s in variants]
    permutation, signs = variants[int(np.argmax(objectives))]
    permutation, signs = np.array(permutation), np.array(signs)
    return scores[:, permutation] * signs, permutation, signs


def colors(scores):
    return torch.sigmoid(2 * torch.from_numpy(scores)).numpy()


@torch.inference_mode()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True, help="JSON model manifest; include heads=0 for alignment")
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--images", type=Path, nargs="+", help="Image files or folders (searched recursively)")
    selection.add_argument("--samples", type=Path, help="JSON image manifest")
    parser.add_argument("--imagenet-root", type=Path)
    parser.add_argument("--imagenet21k-root", type=Path)
    parser.add_argument("--group", default="all", help="Select an image group in --samples (default: all)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    specs = model_specs(args.models)
    if int(specs[0]["heads"]) != 0:
        parser.error("PCA alignment requires a baseline model with heads=0")
    if args.images:
        if args.group != "all":
            parser.error("--group requires --samples")
        images = [{"name": f"{index:04d}_{path.stem}", "path": str(path), "sha256": sha256(path)}
                  for index, path in enumerate(image_paths(args.images), start=1)]
        manifest = {"images": images}
    else:
        manifest = read_json(args.samples)
        images = [row for row in manifest["images"] if args.group == "all" or row.get("group") == args.group]
    if not images:
        parser.error("No images selected")
    names = [row["name"] for row in images]
    if len(names) != len(set(names)):
        parser.error("Selected image names must be unique")
    roots = {"imagenet": args.imagenet_root, "imagenet21k": args.imagenet21k_root}
    paths = []
    for row in images:
        if "dataset" in row:
            if roots.get(row["dataset"]) is None:
                parser.error(f"Provide --{row['dataset']}-root for {row['name']}")
            paths.append(roots[row["dataset"]] / row["path"])
        else:
            paths.append((args.samples.parent if args.samples else Path()) / row["path"])
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline, models, records = {}, [], []
    for spec in specs:
        model, info = load_teacher(spec, args.checkpoint_root, device)
        models.append(info)
        for row, path in zip(images, paths):
            if row.get("sha256") and sha256(path) != row["sha256"]:
                raise ValueError(f"Image hash mismatch: {path}")
            with Image.open(path) as image:
                pixels, grid = resize_image(image.convert("RGB"))
            normalized = TF.normalize(pixels, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            tokens = model.forward_features(normalized[None].to(device))["x_norm_patchtokens"][0]
            scores, pca = project(tokens.cpu().double().numpy())
            heads = int(spec["heads"])
            if heads == 0:
                baseline[row["name"]] = scores
            aligned, permutation, signs = align(baseline[row["name"]], scores)
            rgb = colors(aligned)
            directory = args.output_dir / "maps" / row["name"]
            directory.mkdir(parents=True, exist_ok=True)
            if heads == 0:
                TF.to_pil_image(pixels).save(directory / "original.png")
            patch_colors = (rgb.reshape(*grid, 3) * 255).astype(np.uint8)
            Image.fromarray(np.repeat(np.repeat(patch_colors, 16, axis=0), 16, axis=1)).save(directory / f"{heads}L.png")
            np.savez_compressed(directory / f"{heads}L.npz", scores=aligned, rgb=rgb, grid=grid,
                                basis=pca.components_.T[:, permutation] * signs, mean=pca.mean_,
                                eigenvalues=pca.explained_variance_, permutation=permutation, signs=signs)
            records.append({"image": row["name"], "heads": heads, "grid": list(grid),
                            "permutation": permutation.tolist(), "signs": signs.tolist(), "solver": pca._fit_svd_solver})
            print(f"{heads}L: {row['name']}, {grid[0]} x {grid[1]} patches", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    names = {row["name"] for row in images}
    groups = [{**group, "rows": [name for name in group["rows"] if name in names]}
              for group in manifest.get("groups", [{"name": "images", "rows": [row["name"] for row in images]}])]
    write_json(args.output_dir / "metadata.json", {"measurement": "pca", "models": models, "images": images,
        "groups": [group for group in groups if group["rows"]], "records": records,
        "representation": "final LayerNorm patches, centered PCA, unit sample variance",
        "alignment": "signed permutation maximizing mean Pearson correlation to baseline per image",
        "colors": "sigmoid(2Z), all patches", "inference_precision": "float32", "torch": torch.__version__})


if __name__ == "__main__":
    main()
