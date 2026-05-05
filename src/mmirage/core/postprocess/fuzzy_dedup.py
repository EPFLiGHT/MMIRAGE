"""Fuzzy deduplication for MMIRAGE datasets.

Uses character n-gram MinHash + LSH to identify near-duplicate text samples
and drop them in a streaming "first-seen wins" pass.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Set

from datasets import Dataset

from mmirage.config.config import DeduplicationParams
from mmirage.core.postprocess._text import normalize

logger = logging.getLogger(__name__)


def _check_dependencies() -> None:
    try:
        import datasketch  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Fuzzy deduplication requires `datasketch`. "
            "Install with: pip install 'mmirage[dedup]'"
        ) from e


def _shingles(text: str, k: int) -> Set[bytes]:
    text = normalize(text)
    if len(text) < k:
        return {text.encode("utf-8")}
    return {text[i : i + k].encode("utf-8") for i in range(len(text) - k + 1)}


def deduplicate(dataset: Dataset, params: DeduplicationParams) -> Dataset:
    """Remove near-duplicate samples from a dataset using char-ngram MinHash + LSH.

    Algorithm:
    1. For each row, build the set of character n-grams of size `shingle_size`.
    2. Compute a MinHash signature with `num_perm` permutations.
    3. Query an LSH index built so far. If any near-duplicate is already
       indexed (Jaccard similarity above `threshold`), drop this row.
    4. Otherwise, insert the signature and keep the row.

    Args:
        dataset: HuggingFace Dataset to deduplicate.
        params: Deduplication configuration parameters.

    Returns:
        Filtered Dataset with near-duplicates removed.
    """
    _check_dependencies()
    from datasketch import MinHash, MinHashLSH

    n = len(dataset)
    if n <= 1:
        logger.debug("Dataset has %d row(s), skipping fuzzy dedup.", n)
        return dataset

    if params.text_field not in dataset.column_names:
        raise ValueError(
            f"Text field {params.text_field!r} not in dataset columns: "
            f"{dataset.column_names}"
        )

    lsh = MinHashLSH(threshold=params.threshold, num_perm=params.num_perm)
    keep: List[int] = []
    texts: Iterable = dataset[params.text_field]

    for i, raw in enumerate(texts):
        text = raw if isinstance(raw, str) else str(raw)
        m = MinHash(num_perm=params.num_perm)
        m.update_batch(list(_shingles(text, params.shingle_size)))
        if not lsh.query(m):
            lsh.insert(str(i), m)
            keep.append(i)

    logger.debug(
        "Fuzzy dedup: %d → %d rows (%d duplicates removed).",
        n,
        len(keep),
        n - len(keep),
    )

    return dataset.select(keep)
