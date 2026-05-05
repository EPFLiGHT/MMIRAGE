"""Exact deduplication via blake2b digest of normalized text."""

from __future__ import annotations

import hashlib
import logging
from typing import List, Set

from datasets import Dataset

from mmirage.config.config import DeduplicationParams
from mmirage.core.postprocess._text import normalize

logger = logging.getLogger(__name__)


def exact_deduplicate(dataset: Dataset, params: DeduplicationParams) -> Dataset:
    """Drop rows whose normalized text has been seen before (first-seen wins)."""
    n = len(dataset)
    if n <= 1:
        logger.debug("Dataset has %d row(s), skipping exact dedup.", n)
        return dataset

    if params.text_field not in dataset.column_names:
        raise ValueError(
            f"Text field {params.text_field!r} not in dataset columns: "
            f"{dataset.column_names}"
        )

    seen: Set[bytes] = set()
    keep: List[int] = []

    for i, raw in enumerate(dataset[params.text_field]):
        text = raw if isinstance(raw, str) else str(raw)
        # 16 bytes is collision-free in practice for any realistic dataset size.
        h = hashlib.blake2b(normalize(text).encode("utf-8"), digest_size=16).digest()
        if h not in seen:
            seen.add(h)
            keep.append(i)

    logger.debug(
        "Exact dedup: %d → %d rows (%d duplicates removed).",
        n,
        len(keep),
        n - len(keep),
    )

    return dataset.select(keep)
