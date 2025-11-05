#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import argparse
import hashlib
from typing import Dict, Any, List, Iterable, Optional

from tqdm import tqdm
from json_repair import repair_json
import yaml
import nest_asyncio

nest_asyncio.apply()

import sglang as sgl
from sglang.utils import stream_and_merge
from datasets import load_from_disk

from prompts import ASSISTANT_MD_ENHANCE_PROMPT


# -------------------------
# Minimal helpers
# -------------------------
def hash_to_shard(key: str, num_shards: int) -> int:
    h = int(hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest(), 16)
    return h % num_shards


def iter_all_cases(paths: List[str]) -> Iterable[Dict[str, Any]]:
    for p in paths:
        ds = load_from_disk(p)  # memory-mapped Arrow
        for row in ds:
            yield row


def parse_json_array_safely(text: str) -> Optional[List[Dict[str, Any]]]:
    """
    Extract the outermost JSON array from model output and ensure it's
    a list of dicts with 'role' and 'content'.
    """
    try:
        start = text.find("[")
        end = text.rfind("]") + 1
        if start == -1 or end <= 0:
            return None
        fixed = repair_json(text[start:end])
        arr = json.loads(fixed)
        if not isinstance(arr, list):
            return None
        # light validation
        for item in arr:
            if not (isinstance(item, dict) and "role" in item and "content" in item):
                return None
        return arr
    except Exception:
        return None


def load_engine_from_yaml(config_path: str, *, fallback_model: Optional[str] = None):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    engine_args = dict(cfg.get("engine", {}))
    sampling_params = dict(cfg.get("sampling_params", {}))

    if "model_path" not in engine_args and fallback_model:
        engine_args["model_path"] = fallback_model

    # Best-effort sanity check for tensor_parallel_size
    try:
        import torch
        n_gpus = torch.cuda.device_count()
        tps = int(engine_args.get("tensor_parallel_size", 1))
        if n_gpus and tps > n_gpus:
            print(
                f"[warn] tensor_parallel_size={tps} > available GPUs={n_gpus}. "
                f"Reducing to {n_gpus}.",
                file=sys.stderr,
            )
            engine_args["tensor_parallel_size"] = n_gpus
    except Exception:
        pass

    llm = sgl.Engine(**engine_args)
    return llm, sampling_params


def run_model(llm: sgl.Engine, sampling_params: Dict[str, Any], prompt: str,
              override_max_new_tokens: Optional[int] = None) -> str:
    sp = dict(sampling_params)
    if override_max_new_tokens is not None:
        sp["max_new_tokens"] = int(override_max_new_tokens)
    return stream_and_merge(llm, prompt, sp)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Rewrite only assistant messages into Markdown, preserving conversation structure.")
    ap.add_argument("--datasets", nargs="+", required=True, help="Paths to HF datasets saved with load_from_disk().")
    ap.add_argument("--output_dir", required=True, help="Directory to write sharded JSONL outputs.")
    ap.add_argument("--shard_id", type=int, required=True, help="Shard id (0..num_shards-1).")
    ap.add_argument("--num_shards", type=int, required=True, help="Total number of shards.")
    ap.add_argument("--config", default="config-sglang.yaml", help="YAML config for SGLang engine and sampling params.")
    ap.add_argument("--resume", action="store_true", help="Skip case_ids already processed in the shard file.")
    ap.add_argument("--write_every", type=int, default=100, help="Flush buffer every N rows.")
    ap.add_argument("--max_new_tokens", type=int, default=None, help="Optional override for YAML max_new_tokens.")
    ap.add_argument("--id_field", default="case_id", help="Field name for unique id (fallbacks: 'id').")
    args = ap.parse_args()

    assert 0 <= args.shard_id < args.num_shards, "shard_id must be in [0, num_shards)."
    os.makedirs(args.output_dir, exist_ok=True)
    shard_path = os.path.join(args.output_dir, f"conversations_clean_{args.shard_id}.jsonl")
    err_path   = os.path.join(args.output_dir, f"errors_{args.shard_id}.jsonl")

    # Resume set
    done_ids = set()
    if args.resume and os.path.exists(shard_path):
        with open(shard_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if "case_id" in obj:
                        done_ids.add(str(obj["case_id"]))
                    elif "id" in obj:
                        done_ids.add(str(obj["id"]))
                except Exception:
                    pass

    # Load engine + params
    llm, sampling_params = load_engine_from_yaml(args.config, fallback_model="Qwen/Qwen3-Next-80B-A3B-Instruct")

    buf: List[Dict[str, Any]] = []
    total_seen = 0
    total_kept = 0

    try:
        with open(shard_path, "a", encoding="utf-8") as out_f, \
             open(err_path, "a", encoding="utf-8") as err_f:

            for row in tqdm(iter_all_cases(args.datasets),
                            desc=f"Shard {args.shard_id}/{args.num_shards-1}",
                            dynamic_ncols=True):
                total_seen += 1

                # id + sharding
                case_id = str(row.get(args.id_field) or row.get("id") or f"row_{total_seen}")
                if hash_to_shard(case_id, args.num_shards) != args.shard_id:
                    continue
                if args.resume and case_id in done_ids:
                    continue

                # conversations column
                conversations = row.get("conversations")
                if conversations is None:
                    # If some datasets store it as a string, try to parse
                    text_conv = row.get("text") or row.get("conversation") or None
                    try:
                        conversations = json.loads(text_conv) if text_conv else None
                    except Exception:
                        conversations = None

                if not isinstance(conversations, list) or not conversations:
                    err_f.write(json.dumps({"case_id": case_id, "error": "missing_or_invalid_conversations"}) + "\n")
                    continue

                # Build prompt
                prompt = ASSISTANT_MD_ENHANCE_PROMPT.format(
                    conversation_json=json.dumps(conversations, ensure_ascii=False)
                )

                # Inference (with light retry)
                tries = 0
                transformed = None
                while tries < 3:
                    tries += 1
                    try:
                        raw = run_model(llm, sampling_params, prompt, override_max_new_tokens=args.max_new_tokens)
                        arr = parse_json_array_safely(raw)
                        if arr is not None and len(arr) == len(conversations):
                            transformed = arr
                            break
                        # If parsing failed, retry once or twice
                        time.sleep(1.0 * tries)
                    except Exception as e:
                        if tries >= 3:
                            err_f.write(json.dumps({"case_id": case_id, "error": repr(e)}) + "\n")

                if transformed is None:
                    err_f.write(json.dumps({"case_id": case_id, "error": "parse_failed"}) + "\n")
                    continue

                out_obj = {
                    "case_id": case_id,
                    "conversations": transformed
                }
                buf.append(out_obj)

                if len(buf) >= args.write_every:
                    for r in buf:
                        out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    out_f.flush()
                    total_kept += len(buf)
                    buf.clear()

            # final flush
            if buf:
                for r in buf:
                    out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                out_f.flush()
                total_kept += len(buf)

    except KeyboardInterrupt:
        print("\n[info] Interrupted by user.", file=sys.stderr)
    finally:
        try:
            llm.shutdown()
        except Exception:
            pass

    print(f"✅ Shard {args.shard_id} done. kept={total_kept}, seen={total_seen}, out={shard_path}, err={err_path}")


if __name__ == "__main__":
    main()
