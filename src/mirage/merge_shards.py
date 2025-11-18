import os
import argparse
from typing import List

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

    shard_paths: List[str] = []
    for i in range(args.num_shards):
        shard_dir = os.path.join(args.shards_root, f"shard_{i}")
        if not os.path.isdir(shard_dir):
            raise FileNotFoundError(f"Expected shard directory not found: {shard_dir}")
        shard_paths.append(shard_dir)

    # Load each shard dataset
    shard_dsets = [load_from_disk(p) for p in shard_paths]

    # Concatenate into a single dataset
    ds_merged = concatenate_datasets(shard_dsets)

    # Save final merged dataset
    os.makedirs(args.output_dir, exist_ok=True)
    ds_merged.save_to_disk(args.output_dir)

    print(
        f"✅ Merged {args.num_shards} shards from {args.shards_root} "
        f"into {args.output_dir} with {len(ds_merged)} rows."
    )

if __name__ == "__main__":
    main()