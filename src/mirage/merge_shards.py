"""Script to merge processed dataset shards."""

import argparse
import os

from datasets import concatenate_datasets, load_from_disk


def main():
    """Merge processed shard datasets into a single Hugging Face dataset.

    Loads multiple shard datasets from disk, concatenates them, and saves
    the merged dataset to the specified output directory. Skips invalid
    or empty shards with warnings.
    """
    ap = argparse.ArgumentParser("Merge processed shard datasets into one HF dataset.")
    ap.add_argument(
        "--shards_root",
        required=True,
        help="Directory containing shard_* subdirectories (one per shard).",
    )
    ap.add_argument(
        "--num_shards",
        type=int,
        required=True,
        help="Number of shards you processed (should match the sbatch array size).",
    )
    ap.add_argument(
        "--output_dir",
        required=True,
        help="Base output directory for the merged HF dataset.",
    )
    ap.add_argument(
        "--split",
        type=int,
        default=1,
        help=(
            "Number of splits to produce from the merged dataset. "
            "1 = no split (single dataset at --output_dir). "
            "N>=2 = save N roughly equal splits as <output_dir>_1 ... <output_dir>_N."
        ),
    )
    args = ap.parse_args()

    if args.split < 1:
        raise ValueError("--split must be >= 1")

    shard_dsets = []
    skipped_empty_dir = 0
    skipped_zero_rows = 0

    for i in range(args.num_shards):
        shard_dir = os.path.join(args.shards_root, f"shard_{i}")

        try:
            ds = load_from_disk(shard_dir)
        except FileNotFoundError as e:
            print(
                f"⚠️ {shard_dir} is not a valid HF dataset directory, skipping. "
                f"Reason: {e}"
            )
            skipped_empty_dir += 1
            continue

        if len(ds) == 0:
            print(f"⚠️ Shard dataset has 0 rows, skipping: {shard_dir}")
            skipped_zero_rows += 1
            continue

        print(f"✅ Using shard_{i} with {len(ds)} rows.")
        shard_dsets.append(ds)

    if not shard_dsets:
        raise RuntimeError(
            f"No non-empty shards found in {args.shards_root}. "
            f"empty/invalid dirs: {skipped_empty_dir}, "
            f"zero-row datasets: {skipped_zero_rows}."
        )

    ds_merged = concatenate_datasets(shard_dsets)
    n_rows = len(ds_merged)

    total_skipped = skipped_empty_dir + skipped_zero_rows

    ds_merged.save_to_disk(args.output_dir)

    print(
        f"✅ Concatenated {len(shard_dsets)} shards into a dataset with {n_rows} rows.\n"
        f"   Skipped shards: {total_skipped} total "
        f"(empty/invalid dir: {skipped_empty_dir}, zero rows: {skipped_zero_rows})."
    )


if __name__ == "__main__":
    main()
