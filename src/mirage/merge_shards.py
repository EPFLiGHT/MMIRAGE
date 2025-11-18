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
        help="Output directory where the merged HF dataset will be saved.",
    )
    args = ap.parse_args()

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

    # Save final merged dataset
    os.makedirs(args.output_dir, exist_ok=True)
    ds_merged.save_to_disk(args.output_dir)

    total_skipped = skipped_empty_dir + skipped_zero_rows

    print(
        f"✅ Merged {len(shard_dsets)} shards with data from {args.shards_root} "
        f"into {args.output_dir} with {len(ds_merged)} rows.\n"
        f"   Skipped shards: {total_skipped} total "
        f"empty/invalid dir: {skipped_empty_dir}, "
        f"zero rows: {skipped_zero_rows})."
    )

if __name__ == "__main__":
    main()