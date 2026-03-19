"""Command-line interface for MMIRAGE pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict
from typing import List, Optional

from mmirage.cli_utils.runtime import setup_runtime, validate_paths
from mmirage.cli_utils.slurm import require_slurm, submit_slurm_job, wait_for_slurm_job
from mmirage.cli_utils.status import (
    check_failed_shards,
    status_exit_code,
    submit_failed_shards,
)
from mmirage.config.config import MMirageConfig
from mmirage.config.utils import load_mmirage_config


logger = logging.getLogger(__name__)


def run_local(config_path: str, shard_id: Optional[int] = None) -> int:
    """Run one shard in the current Python environment."""
    command = [sys.executable, "-m", "mmirage.shard_process", "--config", config_path]
    env = os.environ.copy()
    if shard_id is not None:
        env["SLURM_ARRAY_TASK_ID"] = str(shard_id)

    logger.info("Running local shard processing: %s", " ".join(command))
    result = subprocess.run(command, env=env, check=False)
    return result.returncode


def launch_pipeline(cfg: MMirageConfig, config_path: str, force_retry: bool = False) -> int:
    """Launch the pipeline according to execution mode and retry settings."""
    auto_retry = force_retry or cfg.execution_params.retry

    if not cfg.execution_params.is_slurm():
        initial_shard_id = cfg.loading_params.get_shard_id()
        if not auto_retry:
            return run_local(config_path, initial_shard_id)

        if not cfg.loading_params.get_state_root():
            logger.warning(
                "Local retry requires loading_params.state_dir; running once without orchestration"
            )
            return run_local(config_path, initial_shard_id)

        shard_ids: List[int] = [initial_shard_id]
        while True:
            for shard_id in shard_ids:
                run_local(config_path, shard_id)

            failed_shards, summary = check_failed_shards(cfg)
            if status_exit_code(failed_shards, summary) == 0:
                logger.info("All shards completed successfully")
                return 0

            if not failed_shards:
                logger.error("Pipeline ended with shards that exceeded max retries")
                return 1

            logger.warning("Retrying failed shards locally: %s", ",".join(map(str, failed_shards)))
            shard_ids = failed_shards

    shard_ids: List[int] = []

    while True:
        job_id = submit_slurm_job(cfg, config_path, shard_ids)
        if job_id is None:
            return 1

        print(job_id)

        if not auto_retry:
            return 0

        wait_for_slurm_job(job_id, cfg)
        failed_shards, summary = check_failed_shards(cfg)

        if status_exit_code(failed_shards, summary) == 0:
            logger.info("All shards completed successfully")
            return 0

        if not failed_shards:
            logger.error("Pipeline ended with shards that exceeded max retries")
            return 1

        logger.warning("Retrying failed shards: %s", ",".join(map(str, failed_shards)))
        shard_ids = failed_shards


def configure_logging(level: str) -> None:
    """Configure root logging."""
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach common CLI arguments to a subcommand parser."""
    parser.add_argument("--config", required=True, help="Path to a MMIRAGE YAML config file")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity",
    )


def build_argparser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="MMIRAGE command-line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit", help="Submit one SLURM array job")
    add_shared_arguments(submit_parser)
    submit_parser.add_argument(
        "--shard-ids",
        help="Comma-separated shard ids to submit instead of the full array",
    )
    submit_parser.add_argument("--wait", action="store_true", help="Wait for the submitted job")

    check_parser = subparsers.add_parser("check", help="Inspect shard status")
    add_shared_arguments(check_parser)
    check_parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print status summary; do not submit retries.",
    )
    check_retry_group = check_parser.add_mutually_exclusive_group()
    check_retry_group.add_argument(
        "--retry",
        dest="retry",
        action="store_true",
        help="Submit a retry job for failed shards (default unless --summary-only).",
    )
    check_retry_group.add_argument(
        "--no-retry",
        dest="retry",
        action="store_false",
        help="Do not submit retries (same as --summary-only).",
    )
    check_parser.set_defaults(retry=True)
    check_interactive_group = check_parser.add_mutually_exclusive_group()
    check_interactive_group.add_argument(
        "--interactive",
        dest="interactive",
        action="store_true",
        help="Prompt before submitting retry jobs (default).",
    )
    check_interactive_group.add_argument(
        "--no-interactive",
        dest="interactive",
        action="store_false",
        help="Submit retry jobs without prompting.",
    )
    check_parser.set_defaults(interactive=True)

    retry_parser = subparsers.add_parser("retry", help="Submit only failed shards")
    add_shared_arguments(retry_parser)
    retry_group = retry_parser.add_mutually_exclusive_group()
    retry_group.add_argument("--interactive", dest="interactive", action="store_true")
    retry_group.add_argument("--no-interactive", dest="interactive", action="store_false")
    retry_parser.set_defaults(interactive=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run according to execution_params.mode and execution_params.retry",
    )
    add_shared_arguments(run_parser)
    run_parser.add_argument(
        "--force-retry",
        action="store_true",
        help="Enable retry orchestration even if execution_params.retry is false",
    )
    run_parser.add_argument(
        "--shard-id",
        type=int,
        default=None,
        help="Run a single shard locally (overrides execution mode)",
    )

    return parser


def parse_shard_ids(raw_value: Optional[str], num_shards: Optional[int] = None) -> List[int]:
    """Parse a comma-separated shard id list."""
    if not raw_value:
        return []

    shard_ids: List[int] = []
    for raw_shard_id in raw_value.split(","):
        candidate = raw_shard_id.strip()
        if not candidate:
            continue

        try:
            shard_id = int(candidate)
        except ValueError as exc:
            raise ValueError(f"Invalid shard id {candidate!r}; expected integers") from exc

        if shard_id < 0:
            raise ValueError(f"Invalid shard id {shard_id}; expected non-negative integer")
        if num_shards is not None and shard_id >= num_shards:
            raise ValueError(f"Invalid shard id {shard_id}; expected 0 <= shard_id < {num_shards}")

        shard_ids.append(shard_id)

    return shard_ids


def handle_run(args: argparse.Namespace, cfg: MMirageConfig, config_path: str) -> int:
    """Handle the canonical run command."""
    if args.shard_id is not None:
        return run_local(config_path, args.shard_id)
    return launch_pipeline(cfg, config_path, force_retry=args.force_retry)


def handle_submit(args: argparse.Namespace, cfg: MMirageConfig, config_path: str) -> int:
    """Submit a SLURM array job and optionally wait."""
    if require_slurm(cfg, "submit") != 0:
        return 1

    shard_ids = parse_shard_ids(args.shard_ids, cfg.loading_params.get_num_shards())
    job_id = submit_slurm_job(cfg, config_path, shard_ids)
    if job_id is None:
        return 1

    print(job_id)
    if not args.wait:
        return 0

    wait_for_slurm_job(job_id, cfg)
    failed_shards, summary = check_failed_shards(cfg)
    return status_exit_code(failed_shards, summary)


def handle_check(args: argparse.Namespace, cfg: MMirageConfig, config_path: str) -> int:
    """Inspect shard status and optionally submit retries."""
    failed_shards, summary = check_failed_shards(cfg)
    print(json.dumps(asdict(summary), indent=2))

    status_code = status_exit_code(failed_shards, summary)
    if not cfg.execution_params.is_slurm():
        return status_code

    if args.summary_only or not args.retry:
        return status_code

    if not failed_shards:
        return status_code

    return submit_failed_shards(
        cfg=cfg,
        config_path=config_path,
        failed_shards=failed_shards,
        interactive=bool(args.interactive),
    )


def handle_retry(args: argparse.Namespace, cfg: MMirageConfig, config_path: str) -> int:
    """Submit retries for failed shards only."""
    if require_slurm(cfg, "retry") != 0:
        return 1

    failed_shards, summary = check_failed_shards(cfg)
    print(json.dumps(asdict(summary), indent=2))

    if not failed_shards:
        if summary.max_retries_exceeded > 0:
            logger.error("No retryable shards remain")
            return 1
        print("All shards already succeeded.")
        return 0

    return submit_failed_shards(
        cfg=cfg,
        config_path=config_path,
        failed_shards=failed_shards,
        interactive=bool(args.interactive),
    )


def main() -> None:
    """CLI entry point."""
    parser = build_argparser()
    args = parser.parse_args()
    configure_logging(args.log_level)

    try:
        config_path = os.path.abspath(args.config)
        cfg = load_mmirage_config(config_path)

        setup_runtime(cfg, args.log_level)
        validate_paths(cfg)

        handlers = {
            "run": handle_run,
            "submit": handle_submit,
            "check": handle_check,
            "retry": handle_retry,
        }
        handler = handlers.get(args.command)
        if handler is None:
            logger.error("Unknown command: %s", args.command)
            sys.exit(2)

        sys.exit(handler(args, cfg, config_path))

    except Exception as exc:
        logger.error("Error: %s", exc, exc_info=logger.isEnabledFor(logging.DEBUG))
        sys.exit(1)


if __name__ == "__main__":
    main()
