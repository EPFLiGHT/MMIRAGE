"""Local Hugging Face dataset loader implementation."""

from dataclasses import dataclass
from typing import Optional

from datasets import Dataset, load_from_disk, IterableDatasetDict, IterableDataset, DatasetDict, concatenate_datasets
from mirage.core.loader.base import BaseDataLoader, BaseDataLoaderConfig, DataLoaderRegistry

@dataclass
class LocalHFConfig(BaseDataLoaderConfig):
    """Configuration for loading local Hugging Face datasets.

    Attributes:
        type: Type identifier (must be "loadable").
        path: Directory path to the saved Hugging Face dataset.
    """
    path: str = ""


@DataLoaderRegistry.register("loadable", LocalHFConfig)
class LocalHFDataLoader(BaseDataLoader[LocalHFConfig]):
    """Data loader for locally saved Hugging Face datasets.

    Loads datasets from disk that were previously saved using the
    Hugging Face datasets library's save_to_disk method.
    Merges all splits into a single dataset if multiple splits are present.

    Note:
        Iterable datasets are not supported by this loader.
    """

    def from_config(self, ds_config: LocalHFConfig) -> Optional[Dataset]:
        """Load a dataset from a local Hugging Face dataset directory.

        Args:
            ds_config: Configuration containing the path to the dataset directory.

        Returns:
            A Hugging Face Dataset loaded from disk.

        Raises:
            RuntimeError: If the loaded dataset is an iterable dataset.
        """
        ds = load_from_disk(ds_config.path)

        if isinstance(ds, (IterableDatasetDict, IterableDataset)):
            raise RuntimeError(
                f"Iterable datasets are not supported for path: {ds_config.path}"
            )

        if isinstance(ds, DatasetDict):
            ds = concatenate_datasets([ds[split] for split in ds.keys()])

        return ds
