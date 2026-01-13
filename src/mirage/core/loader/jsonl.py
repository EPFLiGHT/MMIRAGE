from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, override
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    concatenate_datasets,
    load_dataset,
)


from mirage.core.loader.base import BaseDataLoader, DataLoaderRegistry, BaseDataLoaderConfig


@dataclass
class JSONLDataConfig(BaseDataLoaderConfig):
    path: str = ""


@DataLoaderRegistry.register("JSONL", JSONLDataConfig)
class JSONLDataLoader(BaseDataLoader[JSONLDataConfig]):
    def __init__(self) -> None:
        super().__init__()
    
    @override
    def from_config(self, ds_config: JSONLDataConfig) -> Optional[Dataset]:
        path = ds_config.path
        ds = load_dataset("json", data_files=path, streaming=False)

        # No support of iterable datasets
        if isinstance(ds, (IterableDatasetDict, IterableDataset)):
            raise RuntimeError(
                f"Iterable datasets are not supported for path: {path}"
            )

        if isinstance(ds, DatasetDict):
            # Merge all splits into one Dataset
            ds = concatenate_datasets([ds[split] for split in ds.keys()])

        return ds

