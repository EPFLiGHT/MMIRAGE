"""Main script for processing dataset shards with MMIRAGE."""

import argparse
from functools import reduce
import os
from typing import Any, Dict, List

from datasets import Dataset, DatasetDict

from mmirage.core.loader.base import BaseDataLoaderConfig, DatasetLike
from mmirage.core.process.mapper import MMIRAGEMapper

from mmirage.config.utils import (
    load_mmirage_config,
)

from mmirage.core.writer.renderer import TemplateRenderer
from mmirage.core.loader.utils import load_datasets_from_configs
import logging
import time

logger = logging.getLogger(__name__)


def _count_rows(ds: DatasetLike) -> int:
    if isinstance(ds, DatasetDict):
        return sum(len(split) for split in ds.values())
    return len(ds)

def _dataset_out_dir(shard_idx: int, ds_config: BaseDataLoaderConfig) -> str:
    return os.path.join(ds_config.output_dir, f"shard_{shard_idx}")


def _shard_dataset(ds: DatasetLike, num_shards: int, shard_id: int) -> DatasetLike:
    if isinstance(ds, DatasetDict):
        return DatasetDict(
            {
                split: split_ds.shard(num_shards=num_shards, index=shard_id)
                for split, split_ds in ds.items()
            }
        )
    return ds.shard(num_shards=num_shards, index=shard_id)


def _remove_columns(ds: DatasetLike, enable: bool) -> List[str]:
    if not enable:
        return []
    if isinstance(ds, DatasetDict):
        columns_set = [set(split_ds.column_names) for split_ds in ds.values()]
        return list(reduce(lambda x, y: x | y, columns_set))
    return ds.column_names


def rewrite_batch(
    batch: Dict[str, List[Any]],
    mapper: MMIRAGEMapper,
    renderer: TemplateRenderer,
) -> Dict[str, List[Any]]:
    """Rewrite a batch of samples by applying transformations.

    Args:
        batch: Dictionary mapping column names to lists of values.
        mapper: MMIRAGEMapper for processing transformations.
        renderer: TemplateRenderer for generating output.

    Returns:
        Dictionary mapping output keys to lists of rendered values.

    Raises:
        ValueError: If variables are not computable given the configuration.
    """
    if not mapper.validate_vars():
        raise ValueError(
            "Uncomputable variables detected. Verify your configuration and make sure that there is no undefined variables"
        )

    batch_environment = mapper.rewrite_batch(batch)
    rendered_list = renderer.batch_render(batch_environment)
    return rendered_list


def main():
    """Process a single shard of the dataset.

    Loads configuration, datasets, processes the shard using MMIRAGE
    transformations, and saves the result to disk.
    """
    ap = argparse.ArgumentParser(
        "Rewrite the assistant turn inside `conversations` into Markdown using SGLang + HF map + sharding."
    )
    ap.add_argument(
        "--config",
        help="YAML config for SGLang engine + sampling + batch_size.",
        required=True,
    )
    args = ap.parse_args()

    cfg = load_mmirage_config(args.config)
    loading_params = cfg.loading_params
    processing_params = cfg.processing_params
    datasets_config = loading_params.datasets
    if not datasets_config:
        raise ValueError("No datasets provided in config.loading_params.datasets")

    shard_id = loading_params.get_shard_id()
    num_shards = loading_params.get_num_shards()

    if not (0 <= shard_id < num_shards):
        raise ValueError(f"Invalid shard_id={shard_id}, num_shards={num_shards}")

    begin_load_ds_time = time.time()
    ds_all = load_datasets_from_configs(datasets_config)
    total_rows = sum(_count_rows(ds) for ds in ds_all)

    ds_all_shard = [_shard_dataset(ds, num_shards, shard_id) for ds in ds_all]
    shard_rows = sum(_count_rows(ds) for ds in ds_all_shard)
    end_load_ds_time = time.time()

    logger.info(
        f"Loaded {len(datasets_config)} dataset(s): {datasets_config} "
        f"→ {total_rows} total rows; this shard has {shard_rows} rows."
    )
    logger.info(f"Dataset loading/sharding time: {end_load_ds_time - begin_load_ds_time}")

    mapper = MMIRAGEMapper(
        cfg.processors, processing_params.inputs, processing_params.outputs
    )
    renderer = TemplateRenderer(processing_params.output_schema)
    ds_processed_all: List[DatasetLike] = []
    begin_process_time = time.time()
    for ds_idx, ds_shard in enumerate(ds_all_shard):
        remove_columns = _remove_columns(ds_shard, processing_params.remove_columns)
        ds_processed = ds_shard.map(
            rewrite_batch,
            batched=True,
            batch_size=loading_params.get_batch_size(),
            load_from_cache_file=False,
            desc=f"Shard {shard_id}/{num_shards - 1} dataset {ds_idx}",
            fn_kwargs={"mapper": mapper, "renderer": renderer},
            remove_columns=remove_columns,
        )
        ds_processed_all.append(ds_processed)
    end_process_time = time.time()
    logger.info(f"Processing time: {end_process_time - begin_process_time}")

    for ds_config, ds_processed in zip(datasets_config, ds_processed_all):
        out_dir = _dataset_out_dir(shard_id, ds_config)
        os.makedirs(out_dir, exist_ok=True)
        ds_processed.save_to_disk(out_dir)

        logger.info(
            f"✅ Saved dataset in: {out_dir} "
        )

if __name__ == "__main__":
    main()
