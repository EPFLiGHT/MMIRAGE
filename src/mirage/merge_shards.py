import os
import argparse

from datasets import load_from_disk, concatenate_datasets


def main():
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

    # Concatenate into a single dataset
    ds_merged = concatenate_datasets(shard_dsets)
    n_rows = len(ds_merged)

    total_skipped = skipped_empty_dir + skipped_zero_rows

    print(
        f"✅ Concatenated {len(shard_dsets)} shards into a dataset with {n_rows} rows.\n"
        f"   Skipped shards: {total_skipped} total "
        f"(empty/invalid dir: {skipped_empty_dir}, zero rows: {skipped_zero_rows})."
    )

    # If only one split requested, just save merged dataset as-is
    if args.split == 1:
        os.makedirs(args.output_dir, exist_ok=True)
        ds_merged.save_to_disk(args.output_dir)
        print(f"💾 Saved merged dataset to {args.output_dir}")
        return

    # --- Split into N roughly equal parts ---
    n_splits = args.split
    if n_rows < n_splits:
        print(
            f"⚠️ Requested --split {n_splits} but dataset has only {n_rows} rows. "
            f"Reducing splits to {n_rows}."
        )
        n_splits = n_rows

    base_output_dir = args.output_dir.rstrip("/")

    saved_splits = 0
    for k in range(n_splits):
        # Even partitioning using integer division
        start = k * n_rows // n_splits
        end = (k + 1) * n_rows // n_splits

        if start >= end:
            print(f"⚠️ Split {k+1} would be empty (start={start}, end={end}), skipping.")
            continue

        ds_part = ds_merged.select(range(start, end))
        out_k = f"{base_output_dir}_{k + 1}"

        os.makedirs(out_k, exist_ok=True)
        ds_part.save_to_disk(out_k)
        print(f"💾 Saved split {k+1}: {len(ds_part)} rows -> {out_k}")
        saved_splits += 1

    if saved_splits == 0:
        raise RuntimeError(
            f"Attempted to split into {n_splits} parts but all splits were empty."
        )

    print(
        f"✅ Finished saving {saved_splits} split(s) from merged dataset "
        f"({n_rows} total rows)."
    )


if __name__ == "__main__":
    main()