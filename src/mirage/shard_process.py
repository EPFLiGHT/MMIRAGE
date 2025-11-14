import os
import sys
import json
import argparse
from typing import Dict, Any, List, Tuple

import yaml
import sglang as sgl
from datasets import load_from_disk

from prompts import ASSISTANT_ONLY_MD_PROMPT


# -------------------------
# helpers
# -------------------------
def load_engine_from_yaml(config_path: str) -> Tuple[sgl.Engine, dict, int]:
    """
    Load SGLang engine, sampling params, and batch size from YAML config.

    Example config:

    engine:
      model_path: "meta-llama/Meta-Llama-3.1-8B-Instruct"

    sampling_params:
      temperature: 0.2
      top_p: 0.9

    batch_size: 64
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    engine_args = dict(cfg.get("engine", {}))
    sampling_params = dict(cfg.get("sampling_params", {}))

    batch_size = int(cfg.get("batch_size", 1) or 1)
    if batch_size < 1:
        batch_size = 1

    llm = sgl.Engine(**engine_args)
    return llm, sampling_params, batch_size


def build_prompt(text: str) -> str:
    """Build the Markdown-enhancement prompt for a single assistant message."""
    payload = json.dumps({"assistant_text": text}, ensure_ascii=False)
    return ASSISTANT_ONLY_MD_PROMPT.format(payload=payload)


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser(
        "Rewrite a single assistant column into Markdown using SGLang + HF map + sharding."
    )
    ap.add_argument(
        "--dataset",
        required=True,
        help="Path to the *original* HF dataset saved with `save_to_disk`.",
    )
    ap.add_argument(
        "--output_dir",
        required=True,
        help="Root directory where shard outputs will be written.",
    )
    ap.add_argument(
        "--num_shards",
        type=int,
        required=True,
        help="Total number of shards (matches your sbatch array size).",
    )
    ap.add_argument(
        "--shard_id",
        type=int,
        required=True,
        help="Index of this shard (0-based; usually $SLURM_ARRAY_TASK_ID).",
    )
    ap.add_argument(
        "--config",
        default="config-sglang.yaml",
        help="YAML config for SGLang engine + sampling + batch_size.",
    )
    ap.add_argument(
        "--assistant_field",
        default="assistant",
        help="Name of the column containing the assistant text to rewrite.",
    )
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=384,
        help="Override max_new_tokens in sampling params.",
    )
    args = ap.parse_args()

    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError(
            f"Invalid shard_id={args.shard_id}, num_shards={args.num_shards}"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    shard_out_dir = os.path.join(args.output_dir, f"shard_{args.shard_id}")
    os.makedirs(shard_out_dir, exist_ok=True)

    # -------------------------
    # Load full dataset and take shard
    # -------------------------
    ds_all = load_from_disk(args.dataset)
    total_rows = len(ds_all)

    ds_shard = ds_all.shard(num_shards=args.num_shards, index=args.shard_id)
    shard_rows = len(ds_shard)

    print(
        f"Loaded dataset from {args.dataset} with {total_rows} rows; "
        f"this shard has {shard_rows} rows."
    )

    assistant_field = args.assistant_field
    if assistant_field not in ds_shard.column_names:
        raise ValueError(
            f"Expected assistant column '{assistant_field}', "
            f"but dataset has columns: {ds_shard.column_names}"
        )

    # -------------------------
    # Load SGLang engine + sampling + batch size
    # -------------------------
    llm, sampling_params, batch_size = load_engine_from_yaml(args.config)

    # Apply script-level override for max_new_tokens
    if args.max_new_tokens is not None:
        sampling_params = dict(sampling_params)
        sampling_params["max_new_tokens"] = int(args.max_new_tokens)

    # -------------------------
    # Batched rewrite function for HF map
    # -------------------------
    def rewrite_batch(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        texts = batch[assistant_field]

        prompts: List[str] = []
        valid_indices: List[int] = []

        # Build prompts only for non-empty texts
        for i, t in enumerate(texts):
            if t is None:
                continue
            s = str(t)
            if not s.strip():
                continue
            prompts.append(build_prompt(s))
            valid_indices.append(i)

        # Nothing to rewrite in this batch
        if not prompts:
            return {assistant_field: texts}

        try:
            # Non-streaming synchronous batch generation
            # See SGLang offline engine docs for llm.generate 
            outputs = llm.generate(prompts, sampling_params)
        except Exception as e:
            print(f"[shard {args.shard_id}] Batch generation failed: {e}", file=sys.stderr)
            # On error, keep original texts for this batch
            return {assistant_field: texts}

        if not isinstance(outputs, list) or len(outputs) != len(prompts):
            print(
                f"[shard {args.shard_id}] Unexpected outputs length from llm.generate: "
                f"expected {len(prompts)}, got {len(outputs) if isinstance(outputs, list) else 'non-list'}",
                file=sys.stderr,
            )
            return {assistant_field: texts}

        # Copy original texts and fill in rewritten Markdown where available
        new_texts = list(texts)
        for out_idx, batch_idx in enumerate(valid_indices):
            out = outputs[out_idx]
            md_text = out.get("text", "")
            new_texts[batch_idx] = md_text

        return {assistant_field: new_texts}

    # -------------------------
    # Apply map with batching
    # -------------------------
    # HF map will call rewrite_batch with batches of size <= batch_size,
    # so with N rows and batch_size=64, llm.generate is called ~N/64 times. 
    ds_processed = ds_shard.map(
        rewrite_batch,
        batched=True,
        batch_size=batch_size,
        load_from_cache_file=False,
        desc=f"Shard {args.shard_id}/{args.num_shards - 1}",
    )

    # -------------------------
    # Save shard as its own HF dataset (all columns preserved)
    # -------------------------
    ds_processed.save_to_disk(shard_out_dir)

    try:
        llm.shutdown()
    except Exception:
        pass

    print(
        f"✅ shard_id={args.shard_id} num_shards={args.num_shards} "
        f"total_rows={total_rows} shard_rows={shard_rows} "
        f"out_dir={shard_out_dir}"
    )


if __name__ == "__main__":
    main()