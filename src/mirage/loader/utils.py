from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, concatenate_datasets, load_from_disk, load_dataset
from typing import List
import os

from mirage.config.generation import DatasetConfig

import logging

logger = logging.getLogger(__name__)

def load_datasets_from_configs(configs: List[DatasetConfig]) -> Dataset:
    valid_ds = []
    for ds_config in configs:
        path = ds_config.path

        if not os.path.exists(path):
            logger.warning(f"⚠️ Dataset path does not exist, skipping: {path}")
            continue
        try:
            if ds_config.type == "JSONL":
                ds = load_dataset("json", data_files=path, streaming=False)
                # no support of iterable datasets
                if isinstance(ds, (IterableDatasetDict, IterableDataset)):
                    raise RuntimeError(
                        f"Iterable datasets are not supported for path: {path}"
                    )
            else:
                ds = load_from_disk(path)

            if isinstance(ds, DatasetDict):
                # Merge all splits into one Dataset
                ds = concatenate_datasets([ds[split] for split in ds.keys()])

            valid_ds.append(ds)
        except Exception as e:
            logger.error(f"⚠️ Failed to load dataset from {path}, skipping. Reason: {e}")

    if not valid_ds:
        raise RuntimeError("No valid datasets loaded from the provided configs.")

    elif len(valid_ds) == 1:
        return valid_ds[0]
    else:
        return concatenate_datasets(valid_ds)


