"""JSONL data loader implementation."""

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
    """Configuration for loading JSONL datasets.

    Attributes:
        type: Type identifier (must be "JSONL").
        path: File path to the JSONL file.
    """
    path: str = ""


@DataLoaderRegistry.register("JSONL", JSONLDataConfig)
class JSONLDataLoader(BaseDataLoader[JSONLDataConfig]):
    """Data loader for JSONL (JSON Lines) formatted datasets.

    Loads datasets from JSONL files using the Hugging Face datasets library.
    Merges all splits into a single dataset if multiple splits are present.

    Note:
        Iterable datasets are not supported by this loader.
    """

    def __init__(self) -> None:
        """Initialize the JSONL data loader."""
        super().__init__()

    @override
    def from_config(self, ds_config: JSONLDataConfig) -> Optional[Dataset]:
        """Load a dataset from a JSONL file.

        Args:
            ds_config: Configuration containing the path to the JSONL file.

        Returns:
            A Hugging Face Dataset containing the JSONL data.

        Raises:
            RuntimeError: If the loaded dataset is an iterable dataset.
        """
        path = ds_config.path
        ds = load_dataset("json", data_files=path, streaming=False)

        if isinstance(ds, (IterableDatasetDict, IterableDataset)):
            raise RuntimeError(
                f"Iterable datasets are not supported for path: {path}"
            )

        if isinstance(ds, DatasetDict):
            ds = concatenate_datasets([ds[split] for split in ds.keys()])

        return ds

