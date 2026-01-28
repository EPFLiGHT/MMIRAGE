"""Main script for processing dataset shards with MMIRAGE."""

import argparse
import os
from typing import Any, Dict, List

from mmirage.core.process.mapper import MMIRAGEMapper

from mmirage.config.utils import (
    load_mmirage_config,
)

from mmirage.core.writer.renderer import TemplateRenderer
from mmirage.core.loader.utils import load_datasets_from_configs
import logging

logger = logging.getLogger(__name__)


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
        default="config-sglang.yaml",
        help="YAML config for SGLang engine + sampling + batch_size.",
    )
    args = ap.parse_args()

    cfg = load_mmirage_config(args.config)
    loading_params = cfg.loading_params
    processing_params = cfg.processing_params
    datasets = loading_params.datasets
    if not datasets:
        raise ValueError("No datasets provided in config.loading_params.datasets")

    shard_id = loading_params.get_shard_id()
    num_shards = loading_params.get_num_shards()

    if not (0 <= shard_id < num_shards):
        raise ValueError(f"Invalid shard_id={shard_id}, num_shards={num_shards}")

    os.makedirs(loading_params.output_dir, exist_ok=True)
    shard_out_dir = os.path.join(loading_params.output_dir, f"shard_{shard_id}")
    os.makedirs(shard_out_dir, exist_ok=True)

    ds_all = load_datasets_from_configs(datasets)
    total_rows = len(ds_all)

    ds_shard = ds_all.shard(num_shards=num_shards, index=shard_id)
    shard_rows = len(ds_shard)

    logger.info(
        f"Loaded {len(datasets)} dataset(s): {datasets} "
        f"→ {total_rows} total rows; this shard has {shard_rows} rows."
    )

    mapper = MMIRAGEMapper(
        cfg.processors, processing_params.inputs, processing_params.outputs
    )
    renderer = TemplateRenderer(processing_params.output_schema)
    ds_processed = ds_shard.map(
        rewrite_batch,
        batched=True,
        batch_size=loading_params.get_batch_size(),
        load_from_cache_file=False,
        desc=f"Shard {shard_id}/{num_shards - 1}",
        fn_kwargs={"mapper": mapper, "renderer": renderer},
        remove_columns=ds_shard.column_names
        if processing_params.remove_columns
        else [],
    )

    ds_processed.save_to_disk(shard_out_dir)

    logger.info(
        f"✅ shard_id={shard_id} num_shards={num_shards} "
        f"total_rows={total_rows} shard_rows={shard_rows} "
        f"out_dir={shard_out_dir}"
    )


if __name__ == "__main__":
    main()
