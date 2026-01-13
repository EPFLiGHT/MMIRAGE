from datasets import Dataset, concatenate_datasets
from typing import List
from mirage.core.loader.base import AutoDataLoader, BaseDataLoaderConfig

import logging

logger = logging.getLogger(__name__)

def load_datasets_from_configs(configs: List[BaseDataLoaderConfig]) -> Dataset:
    valid_ds = []
    for ds_config in configs:
        loader = AutoDataLoader.from_name(ds_config.type)()
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


