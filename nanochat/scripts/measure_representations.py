"""Measure token spectra and all-position next-token ANOVA."""

import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch

from nanochat.measurements.data import load_contexts, sha256_file, write_json
from nanochat.measurements.statistics import ALPHAS, TokenClassMoments, component_counts, spectrum_summary, token_spectrum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Directory containing model_STEP.pt and meta_STEP.json")
    parser.add_argument("--step", type=int, help="Defaults to the latest saved step")
    parser.add_argument("--metrics", choices=("spectrum", "anova", "both"), default="both")
    parser.add_argument("--data", type=Path, required=True, help="Best-fit .npy contexts from prepare_measurement_data")
    parser.add_argument("--output", type=Path, required=True, help="New output .json; spectra are saved alongside as .npz")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--spectrum-sequences", type=int, default=512)
    parser.add_argument("--center", action="store_true", help="Subtract each sequence's mean token vector for spectra only")
    parser.add_argument("--seed", type=int, default=23, help="Spectrum sequence selection seed")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--spectral-device", choices=("cpu", "model"), default="model")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.center and args.metrics == "anova":
        parser.error("--center applies to spectra; use --metrics spectrum or both")
    if min(args.batch_size, args.spectrum_sequences, args.cpu_threads) < 1:
        parser.error("Batch size, sequence count, and thread count must be positive")
    if args.output.suffix != ".json":
        parser.error("Output must end in .json")
    if args.output.exists() or args.output.with_suffix(".npz").exists():
        parser.error("Output already exists; choose a new path")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; --device cpu enables a float32 smoke run")
    os.environ.setdefault("NANOCHAT_DTYPE", "bfloat16" if args.device == "cuda" else "float32")
    from nanochat.common import COMPUTE_DTYPE, get_base_dir
    from nanochat.checkpoint_manager import build_model, find_last_step
    from nanochat.measurements.features import PenultimateMLPInput

    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    step = args.step if args.step is not None else find_last_step(args.checkpoint_dir)
    started = time.monotonic()
    model, _, metadata = build_model(str(args.checkpoint_dir), step, device)
    config = model.config
    tokenizer_hash = sha256_file(Path(get_base_dir()) / "tokenizer/tokenizer.pkl")
    checkpoint = args.checkpoint_dir / f"model_{step:06d}.pt"
    result = {
        "schema_version": 1, "model": metadata["model_tag"], "step": step,
        "depth": config.n_layer, "width": config.n_embd,
        "laplacian_heads": list(config.laplacian_heads), "model_config": config.to_dict(),
        "checkpoint_sha256": sha256_file(checkpoint),
        "tokenizer_sha256": tokenizer_hash,
        "representation": {"location": "penultimate_mlp_input", "block_index": config.n_layer - 2,
                           "normalization": "native RMSNorm after attention residual", "centering": "none"},
        "numerics": {"activation_dtype": str(COMPUTE_DTYPE), "statistics_dtype": "float64",
                     "rms_epsilon": torch.finfo(COMPUTE_DTYPE).eps, "std_ddof": 0,
                     "device": args.device, "spectral_device": args.spectral_device,
                     "batch_size": args.batch_size, "torch": torch.__version__},
    }

    rows, meta = load_contexts(args.data, config.vocab_size, config.sequence_len, tokenizer_hash)
    packing = meta.get("packing", meta.get("sampling"))
    if packing not in ("bestfit", "finite_original_bos_bestfit_four_rank_streams"):
        raise ValueError(f"{args.data}: expected best-fit contexts, found {packing!r}")
    source = {"file": args.data.name, "sha256": meta["sha256"], "packing": "bestfit",
              "num_sequences": len(rows), "sequence_length": config.sequence_len}

    with PenultimateMLPInput(model) as features:
        if args.metrics in ("spectrum", "both"):
            if len(rows) < args.spectrum_sequences:
                raise ValueError("Fewer contexts than --spectrum-sequences; reduce it explicitly for a smoke run")
            indices = np.sort(np.random.default_rng(args.seed).choice(len(rows), args.spectrum_sequences, replace=False))
            eigenvalues = []
            for offset in range(0, len(indices), args.batch_size):
                ids = indices[offset:offset + args.batch_size]
                inputs = torch.as_tensor(np.array(rows[ids, :-1], dtype=np.int64), device=device)
                captured = features(inputs)
                for matrix in captured:
                    eigenvalues.append(token_spectrum(
                        matrix.cpu() if args.spectral_device == "cpu" else matrix, center=args.center,
                    ))
                print(f"Spectrum: {min(offset + len(ids), len(indices))}/{len(indices)} sequences", flush=True)
            values = np.stack(eigenvalues)
            archive = args.output.with_suffix(".npz")
            archive.parent.mkdir(parents=True, exist_ok=True)
            temporary = archive.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, eigenvalues=values, component_counts=component_counts(values),
                                alphas=np.asarray(ALPHAS), sample_indices=indices, centering=args.center)
            temporary.replace(archive)
            result["spectrum"] = {**spectrum_summary(values), "sample": source,
                                  "centering": "per_sequence_token_mean" if args.center else "none",
                                  "sample_indices": indices.tolist(), "selection_seed": args.seed,
                                  "archive": archive.name, "archive_sha256": sha256_file(archive)}
            del captured, eigenvalues, values

        if args.metrics in ("anova", "both"):
            moments = TokenClassMoments(config.vocab_size, config.n_embd, device)
            last_report = time.monotonic()
            for offset in range(0, len(rows), args.batch_size):
                batch = np.array(rows[offset:offset + args.batch_size], dtype=np.int64)
                inputs = torch.as_tensor(batch[:, :-1].copy(), device=device)
                labels = torch.as_tensor(batch[:, 1:].copy(), device=device)
                moments.update(features(inputs), labels)
                done = offset + len(batch)
                if time.monotonic() - last_report >= 20 or done == len(rows):
                    print(f"ANOVA: {done}/{len(rows)} contexts", flush=True)
                    last_report = time.monotonic()
            result["anova"] = {**moments.summary(), "sample": source,
                               "num_contexts": len(rows), "positions_per_context": config.sequence_len,
                               "class_label": "observed next token", "class_filter": "all observed classes"}
    result["elapsed_seconds"] = time.monotonic() - started
    write_json(args.output, result)
    print(f"Saved {args.output} ({result['elapsed_seconds']:.1f}s)")


if __name__ == "__main__":
    main()
