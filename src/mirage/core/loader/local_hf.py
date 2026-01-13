from dataclasses import dataclass
from typing import Optional

from datasets import Dataset, load_from_disk, IterableDatasetDict, IterableDataset, DatasetDict, concatenate_datasets
from mirage.core.loader.base import BaseDataLoader, BaseDataLoaderConfig, DataLoaderRegistry

@dataclass
class LocalHFConfig(BaseDataLoaderConfig):
    path: str = ""


@DataLoaderRegistry.register("loadable", LocalHFConfig)
class LocalHFDataLoader(BaseDataLoader[LocalHFConfig]):
    def from_config(self, ds_config: LocalHFConfig) -> Optional[Dataset]:
        ds = load_from_disk(ds_config.path)

        # No support of iterable datasets
        if isinstance(ds, (IterableDatasetDict, IterableDataset)):
            raise RuntimeError(
                f"Iterable datasets are not supported for path: {ds_config.path}"
            )

        if isinstance(ds, DatasetDict):
            # Merge all splits into one Dataset
            ds = concatenate_datasets([ds[split] for split in ds.keys()])

        return ds
