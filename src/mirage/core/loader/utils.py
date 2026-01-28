"""Utility functions for loading datasets."""

from datasets import Dataset, concatenate_datasets
from typing import List
from mirage.core.loader.base import AutoDataLoader, BaseDataLoaderConfig

import logging

logger = logging.getLogger(__name__)


def load_datasets_from_configs(configs: List[BaseDataLoaderConfig]) -> Dataset:
    """Load and concatenate multiple datasets from configurations.

    Attempts to load datasets using the specified loader configurations.
    Failed loads are logged as warnings and skipped. If multiple datasets
    are successfully loaded, they are concatenated into a single dataset.

    Args:
        configs: List of dataset configuration objects.

    Returns:
        A Hugging Face Dataset containing the combined data from all
        successfully loaded datasets.

    Raises:
        RuntimeError: If no datasets could be loaded successfully.
    """
    
    config_per_type = {}
    for ds_config in configs:
        config_per_type[ds_config.type] = config_per_type.get(ds_config.type, []) + [ds_config]
    
    valid_ds = []
    for config_type, config_list in config_per_type.items():
        loader = AutoDataLoader.from_name(config_type)()
        for ds_config in config_list:
            try:
                ds = loader.from_config(ds_config)
                valid_ds.append(ds)
            except Exception as e:
                logger.warning(f"⚠️ Dataset loading failed with error: {e}. Skipping")

    if not valid_ds:
        raise RuntimeError("No valid datasets loaded from the provided configs.")

    elif len(valid_ds) == 1:
        return valid_ds[0]
    else:
        return concatenate_datasets(valid_ds)
