"""Script to merge processed dataset shards."""

import argparse
import os
from typing import Dict, List

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

from mmirage.core.loader.base import DatasetLike


def _count_rows(ds: DatasetLike) -> int:
    if isinstance(ds, DatasetDict):
        return sum(len(split) for split in ds.values())
    return len(ds)


def _merge_datasetdict(shard_dsets: List[DatasetDict]) -> DatasetDict:
    split_names = sorted({split for ds in shard_dsets for split in ds.keys()})
    merged: Dict[str, Dataset] = {}
    for split in split_names:
        split_dsets = [ds[split] for ds in shard_dsets if split in ds]
        if not split_dsets:
            continue
        merged[str(split)] = concatenate_datasets(split_dsets)
    if not merged:
        raise RuntimeError("All splits were empty after merging.")
    return DatasetDict(merged)


def _merge_shards(shard_dsets: List[DatasetLike]) -> DatasetLike:
    if not shard_dsets:
        raise RuntimeError("No shard datasets to merge.")
    if all(isinstance(ds, DatasetDict) for ds in shard_dsets):
        return _merge_datasetdict([ds for ds in shard_dsets if isinstance(ds, DatasetDict)])
    if any(isinstance(ds, DatasetDict) for ds in shard_dsets):
        raise RuntimeError("Cannot merge mix of Dataset and DatasetDict shards.")
    return concatenate_datasets([ds for ds in shard_dsets if isinstance(ds, Dataset)])


def _list_shard_dirs(dataset_dir: str) -> List[str]:
    shard_dirs: List[str] = []
    for name in os.listdir(dataset_dir):
        if not name.startswith("shard_"):
            continue
        path = os.path.join(dataset_dir, name)
        if os.path.isdir(path):
            shard_dirs.append(path)

    def _shard_key(path: str) -> int:
        base = os.path.basename(path)
        suffix = base.removeprefix("shard_")
        return int(suffix) if suffix.isdigit() else 0

    shard_dirs.sort(key=_shard_key)
    return shard_dirs


def _dataset_dirs(input_dir: str) -> List[str]:
    candidates: List[str] = []
    for name in os.listdir(input_dir):
        path = os.path.join(input_dir, name)
        if not os.path.isdir(path):
            continue
        if _list_shard_dirs(path):
            candidates.append(path)
    return sorted(candidates)


def main():
    """Merge processed shard datasets into per-dataset Hugging Face datasets.

    Scans --input-dir for dataset subdirectories containing shard_* folders.
    For each dataset directory, merges shard datasets and writes to --output-dir
    while preserving the dataset directory name.
    """
    ap = argparse.ArgumentParser("Merge processed shard datasets into HF datasets.")
    ap.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing dataset subdirectories with shard_* folders.",
    )
    ap.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write merged datasets into.",
    )
    args = ap.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    dataset_dirs = _dataset_dirs(input_dir)
    root_shards = _list_shard_dirs(input_dir)

    if not dataset_dirs and root_shards:
        dataset_dirs = [input_dir]

    if not dataset_dirs:
        raise RuntimeError(
            f"No dataset directories with shard_* folders found in {input_dir}."
        )

    for dataset_dir in dataset_dirs:
        shard_dirs = _list_shard_dirs(dataset_dir)
        if not shard_dirs:
            continue

        shard_dsets: List[DatasetLike] = []
        skipped_empty_dir = 0
        skipped_zero_rows = 0

        for shard_dir in shard_dirs:
            try:
                ds = load_from_disk(shard_dir)
            except FileNotFoundError as e:
                print(
                    f"⚠️ {shard_dir} is not a valid HF dataset directory, skipping. "
                    f"Reason: {e}"
                )
                skipped_empty_dir += 1
                continue

            if _count_rows(ds) == 0:
                print(f"⚠️ Shard dataset has 0 rows, skipping: {shard_dir}")
                skipped_zero_rows += 1
                continue

            print(f"✅ Using {os.path.basename(shard_dir)} with {_count_rows(ds)} rows.")
            shard_dsets.append(ds)

        if not shard_dsets:
            raise RuntimeError(
                f"No non-empty shards found in {dataset_dir}. "
                f"empty/invalid dirs: {skipped_empty_dir}, "
                f"zero-row datasets: {skipped_zero_rows}."
            )

        ds_merged = _merge_shards(shard_dsets)
        n_rows = _count_rows(ds_merged)

        total_skipped = skipped_empty_dir + skipped_zero_rows

        if dataset_dir == input_dir:
            ds_out_dir = output_dir
            dataset_name = os.path.basename(os.path.normpath(input_dir))
        else:
            dataset_name = os.path.basename(dataset_dir)
            ds_out_dir = os.path.join(output_dir, dataset_name)

        os.makedirs(ds_out_dir, exist_ok=True)
        ds_merged.save_to_disk(ds_out_dir)

        print(
            f"✅ Concatenated {len(shard_dsets)} shards for {dataset_name} "
            f"with {n_rows} rows.\n"
            f"   Skipped shards: {total_skipped} total "
            f"(empty/invalid dir: {skipped_empty_dir}, zero rows: {skipped_zero_rows})."
        )


if __name__ == "__main__":
    main()
