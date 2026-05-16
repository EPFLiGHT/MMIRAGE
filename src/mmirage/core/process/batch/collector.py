"""Collect provider batch receipts and merge completed rows by source index.

The receiver consumes one or more metadata receipt files, resolves the provider
configuration for each recorded batch, fetches the provider results, and writes a
single JSONL file ordered by the original source row index.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

from mmirage.config.batch_provider import BatchProviderConfig
from mmirage.core.process.batch.metadata_paths import resolve_metadata_paths_from_config
from mmirage.core.process.batch.metadata_utils import (
    BatchMetadataRecord,
    _normalize_metadata_paths,
    _read_metadata_records,
)
from mmirage.core.process.batch.provider_resolution import (
    build_all_provider_configs,
    resolve_provider_configs,
)
from mmirage.core.process.batch.registry import BatchAdapterFactory

logger = logging.getLogger(__name__)


def _aggregate_batch_mappings(
    records: Sequence[BatchMetadataRecord],
) -> Dict[Tuple[str, str], Dict[str, int]]:
    """Group source-index mappings by provider and provider batch ID.

    Later receipts for the same provider batch overwrite earlier entries for the
    same custom ID, which keeps the latest parsed mapping authoritative.
    """
    aggregated: Dict[Tuple[str, str], Dict[str, int]] = {}

    for record in records:
        provider = record.provider
        provider_batch_id = record.provider_batch_id
        mapping = record.custom_id_to_source_index

        if not provider or not provider_batch_id or not mapping:
            continue

        key = (provider, provider_batch_id)
        aggregated.setdefault(key, {})

        for custom_id, source_index in mapping.items():
            aggregated[key][str(custom_id)] = source_index

    return aggregated


def collect_and_merge(
    records: Sequence[BatchMetadataRecord],
    provider_configs: Mapping[str, BatchProviderConfig],
    output_path: str,
) -> List[Dict[str, Any]]:
    """Fetch provider outputs and write merged rows in source index order.

    Args:
        records: Parsed receipt metadata containing provider batch references.
        provider_configs: Provider-specific configuration keyed by normalized
            provider name.
        output_path: Destination JSONL path for the merged output.

    Returns:
        The ordered rows that were written to disk.

    Raises:
        ValueError: If a receipt references a provider that cannot be resolved.
    """
    pair_to_mapping = _aggregate_batch_mappings(records)

    adapters: Dict[str, Any] = {}
    pair_to_results: Dict[Tuple[str, str], Sequence[Dict[str, Any]]] = {}

    for provider, provider_batch_id in pair_to_mapping.keys():
        if provider not in provider_configs:
            raise ValueError(f"No provider config found for '{provider}'.")

        if provider not in adapters:
            adapters[provider] = BatchAdapterFactory.from_config(provider_configs[provider])

        pair = (provider, provider_batch_id)
        pair_to_results[pair] = adapters[provider].retrieve_results(
            provider_batch_id=provider_batch_id,
            config=provider_configs[provider],
        )

    indexed_rows: MutableMapping[Tuple[str, str, str], Dict[str, Any]] = {}
    for pair, mapping in pair_to_mapping.items():
        results = pair_to_results.get(pair, [])
        for result_row in results:
            custom_id = str(result_row.get("custom_id", "")).strip()
            if not custom_id or custom_id not in mapping:
                continue
            row_payload = _build_output_payload(result_row, custom_id=custom_id)
            indexed_rows[(pair[0], pair[1], custom_id)] = {
                "source_index": int(mapping[custom_id]),
                "custom_id": custom_id,
                **row_payload,
            }

    # Sort primarily by source_index and secondarily by custom_id to ensure
    # deterministic ordering when multiple rows share the same source_index.
    ordered_rows = sorted(
        indexed_rows.values(), key=lambda row: (row.get("source_index", 0), row.get("custom_id", ""))
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in ordered_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return ordered_rows


def _build_output_payload(result_row: Mapping[str, Any], custom_id: str = "") -> Dict[str, Any]:
    """Convert provider content into the receiver's output schema.

    The collector preserves raw text for opaque generations, but maps structured
    question/answer JSON into a conversation format expected by downstream
    consumers.
    """
    error_message = str(result_row.get("error_message", "")).strip()
    if error_message:
        return {
            "status": str(result_row.get("status", "error") or "error"),
            "error_message": error_message,
        }

    raw_content = _extract_content_string(result_row)
    if not raw_content:
        return {"caption": ""}

    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
        stripped_content = raw_content.lstrip()
        if stripped_content.startswith(("{", "[")):
            logger.warning(
                f"Failed to parse JSON for result row (custom_id={custom_id}). "
                f"Treating as raw text. Content: {raw_content[:100]}"
            )
        return {"caption": raw_content}

    if isinstance(parsed, dict) and ("question" in parsed or "answer" in parsed):
        return {
            "conversations": [
                {
                    "role": "user",
                    "content": str(parsed.get("question", "")),
                },
                {
                    "role": "assistant",
                    "content": str(parsed.get("answer", "")),
                },
            ]
        }

    return {"caption": raw_content}


def _extract_content_string(result_row: Mapping[str, Any]) -> str:
    """Return the generated text payload as a string.

    The collector treats missing content as empty output rather than a hard
    failure so incomplete provider responses do not block the merge.
    """
    return str(result_row.get("generated_text", ""))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect provider batch outputs and merge rows by source index."
    )
    parser.add_argument(
        "--metadata-path",
        nargs="+",
        help=(
            "Path(s) to metadata JSONL receipt file(s). Supports multiple files. "
            "When omitted, uses metadata_output_path from the config batch_provider blocks "
            "and resolves suffixed receipts like '<base>.text.<run>.jsonl'."
        ),
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to write merged JSONL output.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the collector CLI.

    Reads receipt metadata, resolves provider configs from the supplied MMIRAGE
    configuration, and writes the merged JSONL output path passed on the command
    line.
    """
    args = _build_arg_parser().parse_args(argv)
    from mmirage.config.utils import load_mmirage_config

    try:
        cfg = load_mmirage_config(args.config)
        if args.metadata_path:
            metadata_paths = args.metadata_path
        else:
            all_provider_configs = build_all_provider_configs(cfg)
            metadata_paths = [
                config.metadata_output_path
                for config in all_provider_configs.values()
                if config.metadata_output_path
            ]
            metadata_paths = list(dict.fromkeys(metadata_paths))
            if not metadata_paths:
                raise ValueError(
                    "No metadata paths provided and none found in config batch_provider blocks."
                )
            metadata_paths = resolve_metadata_paths_from_config(metadata_paths)

        if not metadata_paths:
            raise ValueError("No metadata paths provided and none found in config batch_provider blocks.")

        records = _read_metadata_records(metadata_paths)
        provider_configs = resolve_provider_configs(records, cfg)

        rows = collect_and_merge(records, provider_configs, args.output_path)
        print(f"Merged {len(rows)} rows and saved to {args.output_path}")
    except ValueError as exc:
        logger.error(str(exc))
        return 1
    except Exception as exc:
        logger.exception("Collector failed")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
