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

import pyarrow as pa
import pyarrow.ipc as ipc


# -------------------------
# Fast, compact prompt (assistant-only)
# -------------------------
ASSISTANT_ONLY_MD_PROMPT = """
You will receive a JSON object with an array "assistant_texts".
Rewrite each element into clear, structured Markdown.
Keep only information present in the original text.
Do not invent new facts. Preserve special tokens like <|reserved_special_token_0|>.
Return ONLY a JSON array of strings in the same order and length.

Input:
{payload}
""".strip()


# -------------------------
# helpers
# -------------------------
def hash_to_shard(key: str, num_shards: int) -> int:
    h = int(hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest(), 16)
    return h % num_shards

def iter_rows(paths: List[str]) -> Iterable[Dict[str, Any]]:
    for p in paths:
        ds = load_from_disk(p)
        for row in ds:
            yield row

def parse_json_string_list(text: str) -> Optional[List[str]]:
    """Extract outermost JSON array and ensure it's a list[str]."""
    try:
        i, j = text.find("["), text.rfind("]") + 1
        if i < 0 or j <= 0: return None
        fixed = repair_json(text[i:j])
        arr = json.loads(fixed)
        if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
            return arr
        return None
    except Exception:
        return None

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
    ap = argparse.ArgumentParser("Fast: rewrite assistant messages only, save Arrow")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_shards", type=int, required=True)
    ap.add_argument("--shard_id", type=int, required=True)
    ap.add_argument("--config", default="config-sglang.yaml")
    ap.add_argument("--id_field", default="case_id")
    ap.add_argument("--max_new_tokens", type=int, default=384)   # small default
    ap.add_argument("--retries", type=int, default=1)            # fast by default
    args = ap.parse_args()

    assert 0 <= args.shard_id < args.num_shards
    os.makedirs(args.output_dir, exist_ok=True)
    out_arrow = os.path.join(args.output_dir, f"conversations_clean_{args.shard_id}.arrow")

    llm, sampling = load_engine_from_yaml(args.config)

    records: List[Dict[str, Any]] = []
    processed = kept = skipped = 0

    for row in iter_rows(args.datasets):
        processed += 1
        rid = str(row.get(args.id_field) or row.get("id") or f"row_{processed}")
        if hash_to_shard(rid, args.num_shards) != args.shard_id:
            continue

        conv = row.get("conversations")
        if not isinstance(conv, list) or not conv:
            continue

        # collect assistant indices/texts (optionally skip ones that already look like md)
        assistant_idx: List[int] = []
        assistant_texts: List[str] = []
        for i, turn in enumerate(conv):
            if isinstance(turn, dict) and turn.get("role") == "assistant":
                txt = str(turn.get("content", ""))
                assistant_idx.append(i)
                assistant_texts.append(txt)

        # nothing to rewrite
        if not assistant_texts:
            records.append({"id": rid, "conversations": conv})
            kept += 1
            continue

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
            except Exception:
                pass
            if attempt < args.retries:
                time.sleep(min(0.5 * attempt, 2.0))

        if rewritten is None:
            skipped += 1
            # keep original conversation if you prefer; here we keep it unchanged
            records.append({"id": rid, "conversations": conv})
            continue

        # splice rewritten assistant texts back into the original conversation
        conv_out = list(conv)
        for idx, new_text in zip(assistant_idx, rewritten):
            # copy to avoid mutating shared objects
            t = dict(conv_out[idx])
            t["content"] = new_text
            conv_out[idx] = t

        records.append({"id": rid, "conversations": conv_out})
        kept += 1

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

    print(f"✅ shard={args.shard_id} kept={kept} processed={processed} skipped={skipped} out={out_arrow}")

if __name__ == "__main__":
    main()
