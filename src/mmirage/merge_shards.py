"""Script to merge processed dataset shards."""

import argparse
import os
import sys
from typing import Dict, List, Set, Tuple

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

from mmirage.core.loader.base import DatasetLike


def _count_rows(ds: DatasetLike) -> int:
    """Count total rows in a dataset or dataset dict."""
    if isinstance(ds, DatasetDict):
        return sum(len(split) for split in ds.values())
    return len(ds)


def _merge_datasetdict(shard_dsets: List[DatasetDict]) -> DatasetDict:
    """Merge multiple DatasetDicts by concatenating each split."""
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
    """Merge shard datasets into a single dataset."""
    if not shard_dsets:
        raise RuntimeError("No shard datasets to merge.")
    if all(isinstance(ds, DatasetDict) for ds in shard_dsets):
        return _merge_datasetdict(
            [ds for ds in shard_dsets if isinstance(ds, DatasetDict)]
        )
    if any(isinstance(ds, DatasetDict) for ds in shard_dsets):
        raise RuntimeError("Cannot merge mix of Dataset and DatasetDict shards.")
    return concatenate_datasets(
        [ds for ds in shard_dsets if isinstance(ds, Dataset)]
    )


def _list_shard_dirs(dataset_dir: str) -> List[str]:
    """List shard directories in a dataset directory."""
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


def _extract_shard_id(shard_path: str) -> int:
    """Extract shard ID from shard directory path."""
    base = os.path.basename(shard_path)
    suffix = base.removeprefix("shard_")
    return int(suffix) if suffix.isdigit() else -1


def _check_shard_success(shard_dir: str) -> bool:
    """Check if a shard completed successfully based on .SUCCESS marker."""
    success_file = os.path.join(shard_dir, ".SUCCESS")
    return os.path.exists(success_file)


def _check_shard_failed(shard_dir: str) -> bool:
    """Check if a shard failed based on .FAILED marker."""
    failed_file = os.path.join(shard_dir, ".FAILED")
    return os.path.exists(failed_file)


def _analyze_shard_status(
    dataset_dir: str, expected_shards: int = None
) -> Tuple[Set[int], Set[int], Set[int], Set[int]]:
    """Analyze status of all shards in a dataset directory.
    
    Args:
        dataset_dir: Path to dataset directory containing shards
        expected_shards: Expected number of shards (for detecting missing ones)
    
    Returns:
        Tuple of (success_ids, failed_ids, incomplete_ids, missing_ids)
    """
    shard_dirs = _list_shard_dirs(dataset_dir)
    
    success_ids: Set[int] = set()
    failed_ids: Set[int] = set()
    incomplete_ids: Set[int] = set()
    
    for shard_dir in shard_dirs:
        shard_id = _extract_shard_id(shard_dir)
        if shard_id < 0:
            continue
            
        if _check_shard_success(shard_dir):
            success_ids.add(shard_id)
        elif _check_shard_failed(shard_dir):
            failed_ids.add(shard_id)
        else:
            incomplete_ids.add(shard_id)
    
    missing_ids: Set[int] = set()
    if expected_shards is not None:
        existing_ids = success_ids | failed_ids | incomplete_ids
        missing_ids = set(range(expected_shards)) - existing_ids
    
    return success_ids, failed_ids, incomplete_ids, missing_ids


def _dataset_dirs(input_dir: str) -> List[str]:
    """Find dataset directories containing shard folders."""
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
    ap.add_argument(
        "--expected-shards",
        type=int,
        default=None,
        help="Expected number of shards (for detecting missing shards).",
    )
    ap.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Fail if any shards are missing, failed, or incomplete.",
    )
    ap.add_argument(
        "--check-markers",
        action="store_true",
        help="Check for .SUCCESS markers and report shard status.",
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
        dataset_name = os.path.basename(dataset_dir)
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}")
        
        shard_dirs = _list_shard_dirs(dataset_dir)
        if not shard_dirs:
            continue

        # Check shard status if requested
        if args.check_markers or args.expected_shards is not None:
            success_ids, failed_ids, incomplete_ids, missing_ids = _analyze_shard_status(
                dataset_dir, args.expected_shards
            )
            
            total_found = len(success_ids) + len(failed_ids) + len(incomplete_ids)
            total_expected = args.expected_shards if args.expected_shards else total_found
            
            print(f"\n📊 Shard Status Report:")
            print(f"   ✅ Successful:  {len(success_ids)} / {total_expected}")
            print(f"   ❌ Failed:      {len(failed_ids)}")
            print(f"   ⚠️  Incomplete:  {len(incomplete_ids)}")
            print(f"   ❓ Missing:     {len(missing_ids)}")
            
            if failed_ids:
                print(f"\n   Failed shard IDs: {sorted(failed_ids)}")
            if incomplete_ids:
                print(f"   Incomplete shard IDs: {sorted(incomplete_ids)}")
            if missing_ids:
                print(f"   Missing shard IDs: {sorted(missing_ids)}")
            
            # Check if we should fail
            has_problems = bool(failed_ids or incomplete_ids or missing_ids)
            if has_problems and args.fail_on_missing:
                print(f"\n❌ ERROR: Found failed/missing/incomplete shards and --fail-on-missing is set")
                sys.exit(1)
            elif has_problems:
                print(f"\n⚠️  WARNING: Some shards are incomplete - merged dataset may be missing data")
                print(f"   Consider running failure detection and relaunching failed shards:")
                print(f"   python src/mmirage/detect_failures.py --input-dir {input_dir} --num-shards {total_expected}")

        shard_dsets: List[DatasetLike] = []
        skipped_empty_dir = 0
        skipped_zero_rows = 0
        skipped_failed = 0

        for shard_dir in shard_dirs:
            # Skip explicitly failed shards if check-markers is enabled
            if args.check_markers and _check_shard_failed(shard_dir):
                shard_id = _extract_shard_id(shard_dir)
                print(f"⚠️ Skipping failed shard {shard_id}: {shard_dir}")
                skipped_failed += 1
                continue
                
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
                f"zero-row datasets: {skipped_zero_rows}, "
                f"failed shards: {skipped_failed}."
            )

        ds_merged = _merge_shards(shard_dsets)
        n_rows = _count_rows(ds_merged)

        total_skipped = skipped_empty_dir + skipped_zero_rows + skipped_failed

        if dataset_dir == input_dir:
            ds_out_dir = output_dir
            dataset_name = os.path.basename(os.path.normpath(input_dir))
        else:
            dataset_name = os.path.basename(dataset_dir)
            ds_out_dir = os.path.join(output_dir, dataset_name)

        os.makedirs(ds_out_dir, exist_ok=True)
        ds_merged.save_to_disk(ds_out_dir)

        print(
            f"\n✅ Concatenated {len(shard_dsets)} shards for {dataset_name} "
            f"with {n_rows} rows."
        )
        if total_skipped > 0:
            print(
                f"   Skipped shards: {total_skipped} total "
                f"(empty/invalid: {skipped_empty_dir}, zero rows: {skipped_zero_rows}, failed: {skipped_failed})"
            )


if __name__ == "__main__":
    main()
