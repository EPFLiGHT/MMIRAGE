"""Receiver-side helper to check provider batch status from metadata receipts.

Designed for CLI use against JSONL receipt files. Skips malformed lines and
missing keys to keep status checks resilient to partial metadata corruption.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Dict, List, Mapping, Sequence, TextIO, Tuple

from mmirage.config.batch_provider import BatchProviderConfig
from mmirage.core.process.batch.adapter import BatchSubmissionResult
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


def extract_unique_provider_batches(
    metadata_records: Sequence[BatchMetadataRecord],
) -> List[Tuple[str, str]]:
    """Return unique ``(provider, provider_batch_id)`` pairs.

    Normalizes provider names to lowercase and ignores records that do not
    provide both keys, preventing accidental calls with incomplete metadata.
    """
    unique_pairs: List[Tuple[str, str]] = []
    seen = set()

    for record in metadata_records:
        provider = record.provider
        provider_batch_id = record.provider_batch_id

        pair = (provider, provider_batch_id)
        if pair in seen:
            continue
        seen.add(pair)
        unique_pairs.append(pair)

    return unique_pairs


def run_status_checker(
    metadata_records: Sequence[BatchMetadataRecord],
    provider_configs: Mapping[str, BatchProviderConfig],
) -> List[BatchSubmissionResult]:
    """Check batch status for each referenced provider batch.

    Prints a per-batch line and a per-provider summary. Providers missing
    from ``provider_configs`` are skipped rather than failing the run so
    partial configurations still yield useful status output.
    """
    results: List[BatchSubmissionResult] = []
    counter: Dict[str, Dict[str, int]] = {}

    for provider, provider_batch_id in extract_unique_provider_batches(metadata_records):
        if provider not in provider_configs:
            logger.warning(f"Skipping batch {provider_batch_id}: no config for provider '{provider}'.")
            provider_counts = counter.setdefault(provider, {})
            provider_counts["skipped"] = provider_counts.get("skipped", 0) + 1

        else:
            config = provider_configs[provider]
            adapter = BatchAdapterFactory.from_config(config)
            result = adapter.check_batch_status(provider_batch_id=provider_batch_id, config=config)
            results.append(result)

            logger.info(f"Batch {provider_batch_id} ({provider}): {result.status}")
            provider_counts = counter.setdefault(provider, {})
            provider_counts[result.status] = provider_counts.get(result.status, 0) + 1

    print("\n------------ Batch status summary ------------")
    for provider, status_counts in counter.items():
        total = sum(status_counts.values())
        print(f"Provider '{provider}' (Total: {total}):")
        for status, count in status_counts.items():
            print(f"  - {status}: {count}/{total}")

    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the status-check entry point."""
    parser = argparse.ArgumentParser(description="Check provider batch statuses from metadata receipts.")
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
        "--config",
        required=True,
        help="Path to the YAML configuration file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point that returns a process-style status code.

    Returns 0 on success or no batches found, and 1 on configuration or
    provider resolution failures.
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
                logger.error("No metadata paths provided and none found in config batch_provider blocks.")
                return 1
            metadata_paths = resolve_metadata_paths_from_config(metadata_paths)

        if not metadata_paths:
            logger.error("No metadata paths provided and none found in config batch_provider blocks.")
            return 1

        records = _read_metadata_records(metadata_paths)
        pairs = extract_unique_provider_batches(records)
        if not pairs:
            logger.info(f"No provider batch IDs found in metadata file(s): {metadata_paths}")
            return 0

        provider_configs = resolve_provider_configs(records, cfg)
        if not provider_configs:
            logger.error("No supported provider configurations could be built from metadata.")
            return 1
        run_status_checker(
            metadata_records=records,
            provider_configs=provider_configs,
        )
    except Exception as exc:
        logger.exception("Status checker failed")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
