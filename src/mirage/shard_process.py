import argparse
import os
from typing import Any, Dict, List

from mirage.core.process.mapper import MIRAGEMapper

from mirage.config.utils import (
    load_mirage_config,
)

from mirage.core.writer.renderer import TemplateRenderer
from mirage.core.loader.utils import load_datasets_from_configs
import logging

logger = logging.getLogger(__name__)

def rewrite_batch(
        batch: Dict[str, List[Any]],
        mapper: MIRAGEMapper,
        renderer: TemplateRenderer,
    ) -> Dict[str, List[Any]]:
    
    if not mapper.validate_vars():
        raise ValueError("Uncomputable variables detected. Verify your configuration and make sure that there is no undefined variables")

    batch_environment = mapper.rewrite_batch(batch)
    rendered_list = renderer.batch_render(batch_environment)
    return rendered_list


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser(
        "Rewrite the assistant turn inside `conversations` into Markdown using SGLang + HF map + sharding."
    )
    ap.add_argument(
        "--config",
        default="config-sglang.yaml",
        help="YAML config for SGLang engine + sampling + batch_size.",
    )
    args = ap.parse_args()

    # -------------------------
    # Load SGLang engine + sampling + batch size
    # -------------------------
    cfg = load_mirage_config(args.config)
    loading_params = cfg.loading_params
    processing_params = cfg.processing_params
    datasets = loading_params.datasets
    if not datasets:
        raise ValueError(
            "No datasets provided in config.processing_gen_params.datasets"
        )

    shard_id = loading_params.get_shard_id()
    num_shards = loading_params.get_num_shards()

    if not (0 <= shard_id < num_shards):
        raise ValueError(f"Invalid shard_id={shard_id}, num_shards={num_shards}")

    os.makedirs(loading_params.output_dir, exist_ok=True)
    shard_out_dir = os.path.join(loading_params.output_dir, f"shard_{shard_id}")
    os.makedirs(shard_out_dir, exist_ok=True)

    # -------------------------
    # Load all input datasets and concatenate
    # -------------------------
    ds_all = load_datasets_from_configs(datasets)
    total_rows = len(ds_all)

    ds_shard = ds_all.shard(num_shards=num_shards, index=shard_id)
    shard_rows = len(ds_shard)

    logger.info(
        f"Loaded {len(datasets)} dataset(s): {datasets} "
        f"→ {total_rows} total rows; this shard has {shard_rows} rows."
    )

    # -------------------------
    # Apply map with batching
    # -------------------------
    # variable_extractor = VariableExtractor(processing_params.inputs)
    mapper = MIRAGEMapper(cfg.processors, processing_params.inputs, processing_params.outputs) 
    renderer = TemplateRenderer(processing_params.output_schema)
    ds_processed = ds_shard.map(
        rewrite_batch,
        batched=True,
        batch_size=loading_params.get_batch_size(),
        load_from_cache_file=False,
        desc=f"Shard {shard_id}/{num_shards - 1}",
        fn_kwargs={
            "mapper" : mapper,
            "renderer": renderer
        },
        remove_columns=ds_shard.column_names if processing_params.remove_columns else []
    )

    # -------------------------
    # Save shard as its own HF dataset (all columns preserved)
    # -------------------------
    ds_processed.save_to_disk(shard_out_dir)

    logger.info(
        f"✅ shard_id={shard_id} num_shards={num_shards} "
        f"total_rows={total_rows} shard_rows={shard_rows} "
        f"out_dir={shard_out_dir}"
    )


if __name__ == "__main__":
    main()
