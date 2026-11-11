import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from nanochat.measurements.finite_bestfit import bestfit_rows


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)


def prepare_contexts(source, tokenizer, output, sequence_length=2048):
    """Write a finite four-stream pass using the pretraining BOS best-fit packing.

    Stream count is fixed independently of the number of measurement devices.
    """
    source, output = Path(source), Path(output)
    if sequence_length < 2:
        raise ValueError("Expected sequence_length >= 2")
    if output.suffix != ".npy":
        raise ValueError("Output must end in .npy")
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(f"A context sample already exists at {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    statistics = []
    bos = tokenizer.get_bos_token_id()
    with pq.ParquetFile(source) as parquet:
        def batches(rank):
            for group in range(rank, parquet.num_row_groups, 4):
                texts = parquet.read_row_group(group, columns=["text"]).column(0).to_pylist()
                for offset in range(0, len(texts), 128):
                    yield tokenizer.encode(texts[offset:offset + 128], prepend=bos, num_threads=4)

        def rows():
            for rank in range(4):
                stats = {}
                yield from bestfit_rows(batches(rank), sequence_length + 1, 1000, stats)
                statistics.append({"rank": rank, **stats})

        # Stream to a temporary raw file, then copy into a standard .npy memmap.
        # No list of all tokenized contexts is held in RAM.
        with tempfile.TemporaryDirectory(dir=output.parent) as directory:
            raw_path = Path(directory) / "contexts.bin"
            count = 0
            with raw_path.open("wb") as handle:
                for row in rows():
                    np.asarray(row, dtype=np.int32).tofile(handle)
                    count += 1
            if count == 0:
                raise ValueError("The shard produced no complete contexts")
            raw = np.memmap(raw_path, dtype=np.int32, mode="r", shape=(count, sequence_length + 1))
            temporary = Path(directory) / output.name
            array = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.int32, shape=raw.shape)
            for offset in range(0, count, 1024):
                array[offset:offset + 1024] = raw[offset:offset + 1024]
            array.flush()
            del raw, array
            os.replace(temporary, output)
        return {
            "schema_version": 1, "packing": "bestfit", "num_sequences": count,
            "sequence_length": sequence_length, "vocab_size": tokenizer.get_vocab_size(),
            "bos_token_id": bos, "source": source.name,
            "source_sha256": sha256_file(source), "sha256": sha256_file(output),
            "documents": parquet.metadata.num_rows, "rank_statistics": statistics,
        }


def load_contexts(path, vocab_size, sequence_length, tokenizer_sha256):
    path = Path(path)
    rows = np.load(path, mmap_mode="r", allow_pickle=False)
    metadata = json.loads(path.with_suffix(".json").read_text())
    if rows.ndim != 2 or len(rows) == 0 or rows.shape[1] != sequence_length + 1:
        raise ValueError("Contexts must have shape (N, checkpoint sequence length + 1)")
    if not np.issubdtype(rows.dtype, np.integer):
        raise ValueError("Contexts must contain integer token IDs")
    for offset in range(0, len(rows), 1024):
        chunk = rows[offset:offset + 1024]
        if chunk.min() < 0 or chunk.max() >= vocab_size:
            raise ValueError("Context token IDs are outside the checkpoint vocabulary")
    if metadata["sha256"] != sha256_file(path):
        raise ValueError("Context checksum differs from its metadata")
    if metadata["tokenizer_sha256"] != tokenizer_sha256:
        raise ValueError("Contexts were prepared with a different tokenizer")
    return rows, metadata
