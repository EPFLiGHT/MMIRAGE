"""Smoke test for deduplication: synthetic offline assertions + TinyStories."""

import argparse
import logging

from datasets import Dataset, load_dataset

from mmirage.config.config import DeduplicationParams
from mmirage.core.postprocess.exact_dedup import exact_deduplicate
from mmirage.core.postprocess.fuzzy_dedup import deduplicate


SYNTHETIC_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "The quick brown fox jumps over the lazy dog.",
    "The quick brown fox jumps over the lazy dogs.",
]


def _params(mode: str, threshold: float, num_perm: int, shingle_size: int) -> DeduplicationParams:
    return DeduplicationParams(
        enabled=True,
        exact=mode in ("exact", "both"),
        fuzzy=mode in ("fuzzy", "both"),
        text_field="text",
        threshold=threshold,
        num_perm=num_perm,
        shingle_size=shingle_size,
    )


def _run_synthetic(mode: str, threshold: float, num_perm: int, shingle_size: int) -> None:
    ds = Dataset.from_dict({"text": SYNTHETIC_TEXTS})
    params = _params(mode, threshold, num_perm, shingle_size)

    after_exact = ds
    if params.exact:
        after_exact = exact_deduplicate(ds, params)
    final = after_exact
    if params.fuzzy:
        final = deduplicate(after_exact, params)

    print(
        f"[synthetic mode={mode}] {len(ds)} → {len(after_exact)} → {len(final)} rows"
    )

    if mode == "exact":
        assert len(final) == 2, f"exact-only should keep 2 rows, got {len(final)}"
    elif mode == "fuzzy":
        assert len(final) == 1, f"fuzzy-only should keep 1 row, got {len(final)}"
    elif mode == "both":
        assert len(after_exact) == 2, f"exact stage should leave 2 rows, got {len(after_exact)}"
        assert len(final) == 1, f"both should end with 1 row, got {len(final)}"

    print(f"[synthetic mode={mode}] OK")


def _run_tinystories(limit, mode, threshold, num_perm, shingle_size):
    ds = load_dataset("roneneldan/TinyStories", split="train")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    print(f"Loaded {len(ds):,} rows")

    params = _params(mode, threshold, num_perm, shingle_size)

    out = ds
    if params.exact:
        out = exact_deduplicate(out, params)
        print(f"After exact: {len(out):,} rows")
    if params.fuzzy:
        out = deduplicate(out, params)
        print(f"After fuzzy: {len(out):,} rows")

    removed = len(ds) - len(out)
    print(
        f"{len(ds):,} → {len(out):,} "
        f"(removed {removed:,}, {removed / len(ds) * 100:.2f}%)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--synthetic",
        action="store_true",
        help="Run the offline synthetic case (no network) and assert behavior.",
    )
    ap.add_argument(
        "--mode",
        choices=["exact", "fuzzy", "both"],
        default="both",
        help="Which dedup stages to run.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for the TinyStories smoke test.",
    )
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--num-perm", type=int, default=128)
    ap.add_argument("--shingle-size", type=int, default=5)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.synthetic:
        _run_synthetic(args.mode, args.threshold, args.num_perm, args.shingle_size)
        return

    _run_tinystories(
        args.limit, args.mode, args.threshold, args.num_perm, args.shingle_size
    )


if __name__ == "__main__":
    main()
