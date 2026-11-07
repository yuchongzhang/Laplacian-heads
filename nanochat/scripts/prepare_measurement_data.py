"""Prepare best-fit contexts shared by spectra and ANOVA."""

import argparse
from pathlib import Path

from nanochat.measurements.data import prepare_contexts, sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Held-out Parquet shard with a text column")
    parser.add_argument("--output", type=Path, required=True, help="New .npy contexts file; metadata is written beside it")
    parser.add_argument("--sequence-length", type=int, default=2048)
    args = parser.parse_args()
    from nanochat.common import get_base_dir
    from nanochat.tokenizer import get_tokenizer
    tokenizer = get_tokenizer()
    tokenizer_hash = sha256_file(Path(get_base_dir()) / "tokenizer/tokenizer.pkl")
    metadata = prepare_contexts(args.input, tokenizer, args.output, args.sequence_length)
    metadata["tokenizer_sha256"] = tokenizer_hash
    write_json(args.output.with_suffix(".json"), metadata)
    print(f"Saved {metadata['num_sequences']:,} contexts to {args.output}")


if __name__ == "__main__":
    main()
