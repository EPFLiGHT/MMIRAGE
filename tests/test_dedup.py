"""Smoke test for fuzzy deduplication on TinyStories."""

import argparse
import logging

from datasets import load_dataset

from mmirage.config.config import DeduplicationParams
from mmirage.core.postprocess.fuzzy_dedup import deduplicate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit (default: full dataset).",
    )
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--num-perm", type=int, default=128)
    ap.add_argument("--shingle-size", type=int, default=5)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)

    ds = load_dataset("roneneldan/TinyStories", split="train")
    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"Loaded {len(ds):,} rows")

    params = DeduplicationParams(
        enabled=True,
        text_field="text",
        threshold=args.threshold,
        num_perm=args.num_perm,
        shingle_size=args.shingle_size,
    )
    deduped = deduplicate(ds, params)
    removed = len(ds) - len(deduped)
    print(
        f"{len(ds):,} → {len(deduped):,} "
        f"(removed {removed:,}, {removed / len(ds) * 100:.2f}%)"
    )


if __name__ == "__main__":
    main()
