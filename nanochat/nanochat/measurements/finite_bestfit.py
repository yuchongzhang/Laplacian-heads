"""The training loader's BOS/best-fit rule, with a finite document source."""


def bestfit_rows(document_batches, capacity, buffer_size=1000, statistics=None):
    """Yield full rows; crop tails as in training and never cycle documents.

    Batches contain already tokenized documents, including their initial BOS.
    At EOF, drain the buffer using the same selection rule. An incomplete final
    row is counted and discarded so every returned row has exactly capacity tokens.
    """
    stats = statistics if statistics is not None else {}
    stats.update(documents=0, input_tokens=0, cropped_tokens=0,
                 incomplete_row_tokens=0, rows=0)
    batches, buffer, exhausted = iter(document_batches), [], False
    row = []
    while True:
        while len(buffer) < buffer_size and not exhausted:
            try:
                batch = next(batches)
            except StopIteration:
                exhausted = True
                break
            for document in batch:
                if not document:
                    raise ValueError("Documents must include a BOS token")
                buffer.append(document)
                stats["documents"] += 1
                stats["input_tokens"] += len(document)
        if not buffer:
            stats["incomplete_row_tokens"] = len(row)
            assert stats["input_tokens"] == (
                stats["rows"] * capacity + stats["cropped_tokens"] + len(row))
            return
        remaining = capacity - len(row)
        best_index, best_length = -1, 0
        for index, document in enumerate(buffer):
            if best_length < len(document) <= remaining:
                best_index, best_length = index, len(document)
        if best_index < 0:
            best_index = min(range(len(buffer)), key=lambda index: len(buffer[index]))
        document = buffer.pop(best_index)
        row.extend(document[:remaining])
        stats["cropped_tokens"] += max(0, len(document) - remaining)
        if len(row) == capacity:
            stats["rows"] += 1
            yield row
            row = []
