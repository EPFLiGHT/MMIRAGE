"""Main script for processing dataset shards with MMIRAGE.

Supports both text-only and multimodal (vision-language) processing.
"""

import argparse
import logging
import os
import sys
import traceback
from typing import Any, Dict, List, Optional

from mmirage.config.utils import load_mmirage_config
from mmirage.core.loader.base import DatasetLike
from mmirage.core.loader.utils import load_datasets_from_configs
from mmirage.core.process.mapper import MMIRAGEMapper
from mmirage.core.writer.renderer import TemplateRenderer
from mmirage.shard_utils import (
    GpuUtilizationPoller,
    ShardStats,
    _cleanup_old_shard_data,
    _count_rows,
    _dataset_out_dir,
    _mark_failure,
    _mark_running,
    _mark_success,
    _remove_columns,
    _save_dataset_atomic,
    _shard_dataset,
    shard_state_dir,
)

logger = logging.getLogger(__name__)


def rewrite_batch(
    batch: Dict[str, List[Any]],
    mapper: MMIRAGEMapper,
    renderer: TemplateRenderer,
    image_base_path: Optional[str] = None,
) -> Dict[str, List[Any]]:
    """Rewrite a batch of samples by applying transformations.
    Args:
        batch: Dictionary mapping column names to lists of values.
        mapper: MMIRAGEMapper for processing transformations.
        renderer: TemplateRenderer for generating output.
        image_base_path: Optional base directory for resolving relative image paths.
    Returns:
        Dictionary mapping output keys to lists of rendered values.
    Raises:
        ValueError: If variables are not computable given the configuration.
    """
    if not mapper.validate_vars():
        raise ValueError(
            "Uncomputable variables detected. Verify your configuration and make sure that there is no undefined variables"
        )

    batch_environment = mapper.rewrite_batch(batch, image_base_path)
    rendered_list = renderer.batch_render(batch_environment)
    return rendered_list


def main():
    """
        Process a single shard of the dataset.
        Loads configuration, datasets, processes the shard using MMIRAGE
        transformations (including multimodal), and saves the result to disk.
    """
    ap = argparse.ArgumentParser("Process dataset shards using MMIRAGE with SGLang.")
    ap.add_argument(
        "--config",
        help="YAML config for MMIRAGE pipeline.",
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
    last_shard_id = num_shards - 1

    if not (0 <= shard_id < num_shards):
        raise ValueError(f"Invalid shard_id={shard_id}, num_shards={num_shards}")

    state_dir = shard_state_dir(shard_id, loading_params.get_state_root())

    gpu_poller: Optional[GpuUtilizationPoller] = None

    collect_stats = os.environ.get("MMIRAGE_COLLECT_STATS", "") == "1"
    if collect_stats:
        # Determine which physical GPU indices SGLang will use so the poller
        # measures only the active GPU(s) — not all GPUs on the node.
        # SLURM may allocate more GPUs than tp_size (e.g. gpus=4, tp_size=1).
        # We take only the first tp_size entries from CUDA_VISIBLE_DEVICES so
        # nvidia-smi --id receives exactly the GPUs SGLang is using.
        tp_size = 1
        for proc_cfg in cfg.processors:
            tp = getattr(getattr(proc_cfg, "server_args", None), "tp_size", None)
            if tp and int(tp) > 0:
                tp_size = int(tp)
                break
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible and cuda_visible.lower() not in ("all", "nodevfiles"):
            all_visible = [x.strip() for x in cuda_visible.split(",") if x.strip()]
            # Fall back to range-based indices if CUDA_VISIBLE_DEVICES was set
            # but contained only whitespace/empty entries after stripping.
            gpu_indices_for_polling: List[str] = all_visible[:tp_size] if all_visible else [str(i) for i in range(tp_size)]
        else:
            gpu_indices_for_polling = [str(i) for i in range(tp_size)]

        gpu_poller = GpuUtilizationPoller(
            interval_seconds=5.0, gpu_indices=gpu_indices_for_polling
        )

    try:
        retry_count = _mark_running(state_dir, shard_id, datasets_config)
        logger.info(f"Starting shard {shard_id}/{last_shard_id} (attempt #{retry_count})")

        if retry_count > 1:
            for ds_config in datasets_config:
                out_dir = _dataset_out_dir(shard_id, ds_config)
                _cleanup_old_shard_data(out_dir)

        ds_all = load_datasets_from_configs(datasets_config)
        total_rows = sum(_count_rows(ds) for ds in ds_all)

        ds_all_shard = [_shard_dataset(ds, num_shards, shard_id) for ds in ds_all]
        shard_rows = sum(_count_rows(ds) for ds in ds_all_shard)

        logger.info(
            f"Loaded {len(datasets_config)} dataset(s): {datasets_config} "
            f"→ {total_rows} total rows; this logical shard has {shard_rows} rows."
        )

        mapper = MMIRAGEMapper(
            cfg.processors,
            processing_params.inputs,
            processing_params.outputs,
        )
        renderer = TemplateRenderer(processing_params.output_schema)

        # Start GPU polling after model loading so utilisation samples reflect
        # inference only, not weight transfers during sgl.Engine() init.
        if collect_stats and gpu_poller is not None:
            gpu_poller.start()

        ds_processed_all: List[DatasetLike] = []
        for ds_idx, ds_shard in enumerate(ds_all_shard):
            ds_config = datasets_config[ds_idx]
            if processing_params.remove_columns:
                remove_columns = _remove_columns(ds_shard)
            else:
                remove_columns = []

            logger.info(
                f"Processing dataset {ds_idx} for shard {shard_id}: "
                f"image_base_path={ds_config.image_base_path}, output_dir={ds_config.output_dir}"
            )

            ds_processed = ds_shard.map(
                rewrite_batch,
                batched=True,
                batch_size=loading_params.get_batch_size(),
                load_from_cache_file=False,
                desc=f"Shard {shard_id}/{last_shard_id} dataset {ds_idx}",
                fn_kwargs={
                    "mapper": mapper,
                    "renderer": renderer,
                    "image_base_path": ds_config.image_base_path,
                },
                remove_columns=remove_columns,
            )
            ds_processed_all.append(ds_processed)

        for ds_idx, (ds_config, ds_processed) in enumerate(zip(datasets_config, ds_processed_all)):
            out_dir = _dataset_out_dir(shard_id, ds_config)
            _save_dataset_atomic(ds_processed, out_dir)
            logger.info(f"✅ Saved dataset {ds_idx} shard in: {out_dir}")

        gpu_info = gpu_poller.stop() if collect_stats and gpu_poller is not None else {"mean": None, "min": None, "max": None, "samples": 0}

        # Collect token counts accumulated by LLM processor(s).
        token_counts = mapper.get_token_counts()
        input_tokens = token_counts.input_tokens or None
        output_tokens = token_counts.output_tokens or None
        model_load_seconds = mapper.get_load_time() or None

        # Resolve num_gpus from the first processor config that exposes tp_size.
        num_gpus: Optional[int] = None
        for proc_cfg in cfg.processors:
            tp = getattr(getattr(proc_cfg, "server_args", None), "tp_size", None)
            if tp and tp > 0:
                num_gpus = int(tp)
                break

        stats = ShardStats(
            rows_processed=shard_rows,
            gpu_util_mean=gpu_info["mean"],
            gpu_util_min=gpu_info["min"],
            gpu_util_max=gpu_info["max"],
            gpu_util_samples=gpu_info["samples"],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            num_gpus=num_gpus,
            model_load_seconds=model_load_seconds,
        )
        _mark_success(state_dir, stats=stats)
        logger.info(f"✅ Logical shard {shard_id} completed successfully")

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"❌ Shard {shard_id} failed: {error_msg}")
        logger.error(traceback.format_exc())
        if collect_stats and gpu_poller is not None:
            gpu_poller.stop()
        _mark_failure(state_dir, error_msg)
        sys.exit(1)


if __name__ == "__main__":
    main()