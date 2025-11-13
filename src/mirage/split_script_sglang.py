#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import argparse
from typing import Dict, Any, List, Optional

from tqdm import tqdm
from json_repair import repair_json
import yaml
import asyncio
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except Exception as e:
    print(f"uvloop import or setup failed: {e}")
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import sglang as sgl
from sglang.utils import stream_and_merge
from datasets import load_from_disk

from prompts import ASSISTANT_ONLY_MD_PROMPT

import pyarrow as pa
import pyarrow.ipc as ipc

# -------------------------
# helpers
# -------------------------
def parse_json_string_list(text: str) -> list[str]:
    """Parse a strict JSON array of strings. Raise on anything else."""
    arr = json.loads(text)
    if not isinstance(arr, list) or not all(isinstance(x, str) for x in arr):
        raise ValueError("Expected JSON array of strings")
    return arr

def load_engine_from_yaml(config_path: str) -> tuple[sgl.Engine, dict]:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    engine_args = dict(cfg.get("engine", {}))
    sampling = dict(cfg.get("sampling_params", {}))
    return sgl.Engine(**engine_args), sampling

def run(llm: sgl.Engine, sampling: dict, prompt: str, max_new_tokens: Optional[int]) -> str:
    sp = dict(sampling)
    if max_new_tokens is not None:
        sp["max_new_tokens"] = int(max_new_tokens)
    return stream_and_merge(llm, prompt, sp)


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser("Rewrite assistant messages only, save Arrow (index-based sharding) with tqdm.")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_shards", type=int, required=True)
    ap.add_argument("--shard_id", type=int, required=True)
    ap.add_argument("--config", default="config-sglang.yaml")
    ap.add_argument("--id_field", default="case_id")  # only for output id
    ap.add_argument("--max_new_tokens", type=int, default=384)   # small/fast default
    ap.add_argument("--retries", type=int, default=1)            # default fast
    args = ap.parse_args()

    assert 0 <= args.shard_id < args.num_shards
    os.makedirs(args.output_dir, exist_ok=True)
    out_arrow = os.path.join(args.output_dir, f"conversations_clean_{args.shard_id}.arrow")

    # Load datasets once so we can compute total rows for tqdm
    dsets = [load_from_disk(p) for p in args.datasets]
    total_rows = sum(len(ds) for ds in dsets)

    llm, sampling = load_engine_from_yaml(args.config)

    records: List[Dict[str, Any]] = []
    kept = skipped = 0

    i_global = 0  # global index across all datasets
    with tqdm(total=total_rows, desc=f"Shard {args.shard_id}/{args.num_shards-1}", dynamic_ncols=True) as pbar:
        for ds in dsets:
            for row in ds:
                # index-based sharding: process only rows where i % num_shards == shard_id
                take = (i_global % args.num_shards) == args.shard_id

                if take:
                    rid = str(row.get(args.id_field) or row.get("id") or f"row_{i_global}")

                    conv = row.get("conversations")
                    if not isinstance(conv, list) or not conv:
                        skipped += 1
                    else:
                        # gather assistant turns
                        assistant_idx: List[int] = []
                        assistant_texts: List[str] = []
                        for ti, turn in enumerate(conv):
                            if isinstance(turn, dict) and turn.get("role") == "assistant":
                                assistant_idx.append(ti)
                                assistant_texts.append(str(turn.get("content", "")))

                        if not assistant_texts:
                            # nothing to rewrite; keep as-is
                            records.append({"id": rid, "conversations": conv})
                            kept += 1
                        else:
                            payload = json.dumps({"assistant_texts": assistant_texts}, ensure_ascii=False)
                            prompt = ASSISTANT_ONLY_MD_PROMPT.format(payload=payload)

                            rewritten: Optional[List[str]] = None
                            for attempt in range(1, args.retries + 1):
                                try:
                                    raw = run(llm, sampling, prompt, args.max_new_tokens)
                                    lst = parse_json_string_list(raw)
                                    if lst is not None and len(lst) == len(assistant_texts):
                                        rewritten = lst
                                        break
                                except Exception as e:
                                    print(f"Attempt {attempt} failed for row id={rid}: {e}")
                                if attempt < args.retries:
                                    time.sleep(min(0.5 * attempt, 2.0))

                            if rewritten is None:
                                # keep original conversation on failure (simple behavior)
                                records.append({"id": rid, "conversations": conv})
                                skipped += 1
                            else:
                                # splice back rewritten assistant messages
                                conv_out = list(conv)
                                for idx, new_text in zip(assistant_idx, rewritten):
                                    t = dict(conv_out[idx])
                                    t["content"] = new_text
                                    conv_out[idx] = t
                                records.append({"id": rid, "conversations": conv_out})
                                kept += 1

                i_global += 1
                pbar.update(1)

    # write Arrow
    convo_struct = pa.struct([pa.field("role", pa.string()), pa.field("content", pa.string())])
    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("conversations", pa.list_(convo_struct)),
    ])
    tbl = pa.Table.from_pylist(records, schema=schema)
    with pa.OSFile(out_arrow, "wb") as sink:
        with ipc.new_file(sink, tbl.schema) as writer:
            writer.write_table(tbl)

    try:
        llm.shutdown()
    except Exception:
        pass

    print(f"✅ shard={args.shard_id} kept={kept} skipped={skipped} total_rows={total_rows} out={out_arrow}")

if __name__ == "__main__":
    main()