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

from mmirage.cli_utils.runtime import setup_runtime, validate_edf_env_path
from mmirage.cli_utils.slurm import require_slurm, submit_slurm_job, wait_for_slurm_job
from mmirage.cli_utils.status import (
    check_failed_shards,
    is_retry_budget_exceeded,
    shard_state_dir,
    get_shard_status,
    status_exit_code,
    submit_failed_shards,
)
from mmirage.config.config import MMirageConfig
from mmirage.config.utils import load_mmirage_config


logger = logging.getLogger(__name__)


def run_local(config_path: str, shard_id: Optional[int] = None) -> int:
    """Run one shard in the current Python environment.

    Args:
        config_path: Absolute path to the MMIRAGE YAML config file.
        shard_id: Optional shard id to inject via SLURM_ARRAY_TASK_ID.

    Returns:
        Process return code from shard execution.
    """
    command = [sys.executable, "-m", "mmirage.shard_process", "--config", config_path]
    env = os.environ.copy()
    if shard_id is not None:
        env["SLURM_ARRAY_TASK_ID"] = str(shard_id)

    logger.info("Running local shard processing: %s", " ".join(command))
    result = subprocess.run(command, env=env, check=False)
    return result.returncode


def launch_pipeline(cfg: MMirageConfig, config_path: str, force_retry: bool = False) -> int:
    """Launch the pipeline according to execution mode and retry settings.

    Args:
        cfg: Parsed MMIRAGE configuration object.
        config_path: Absolute path to the MMIRAGE YAML config file.
        force_retry: If True, enable retry orchestration regardless of config flag.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    auto_retry = force_retry or cfg.execution_params.retry

    if not cfg.execution_params.is_slurm():
        initial_shard_id = cfg.loading_params.get_shard_id()
        if not auto_retry:
            exit_code = run_local(config_path, initial_shard_id)
            if exit_code == 0:
                logger.info("All shards completed successfully")
            return exit_code

        shard_ids: List[int] = [initial_shard_id]
        attempts_by_shard = {initial_shard_id: 0}
        state_root = cfg.loading_params.get_state_root()
        while True:
            run_exit_codes = {}
            for shard_id in shard_ids:
                attempts_by_shard[shard_id] = attempts_by_shard.get(shard_id, 0) + 1
                run_exit_codes[shard_id] = run_local(config_path, shard_id)

            failed_shards, summary = check_failed_shards(cfg)
            if status_exit_code(failed_shards, summary) == 0:
                logger.info("All shards completed successfully")
                return 0

            runtime_failed = [shard_id for shard_id, rc in run_exit_codes.items() if rc != 0]
            candidates = sorted(set(failed_shards) | set(runtime_failed))
            retryable_shards: List[int] = []
            for shard_id in candidates:
                _, state_attempt_count = get_shard_status(shard_state_dir(state_root, shard_id))
                memory_attempt_count = attempts_by_shard.get(shard_id, 0)
                effective_attempt_count = max(state_attempt_count, memory_attempt_count)

                if not is_retry_budget_exceeded(
                    effective_attempt_count,
                    cfg.execution_params.max_retries,
                ):
                    retryable_shards.append(shard_id)

            if not retryable_shards:
                logger.error("Pipeline ended with shards that exceeded max retries")
                return 1

            logger.warning("Retrying failed shards locally: %s", ",".join(map(str, retryable_shards)))
            shard_ids = retryable_shards

    shard_ids: List[int] = []

    while True:
        job_id = submit_slurm_job(cfg, config_path, shard_ids)
        if job_id is None:
            return 1

        logger.info(f"Submitted SLURM job {job_id} for shard ids: {shard_ids or 'ALL'}")

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
    """Configure root logging.

    Args:
        level: Root log level name.
    """
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach common CLI arguments to a subcommand parser.

    Args:
        parser: Subcommand parser receiving shared arguments.
    """
    parser.add_argument("--config", required=True, help="Path to a MMIRAGE YAML config file")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity",
    )


def build_argparser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    Returns:
        Configured top-level argparse parser.
    """
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
        "--retry",
        dest="retry",
        action="store_true",
        help="Submit a retry job for failed shards.",
    )
    check_parser.set_defaults(retry=False)
    check_parser.add_argument(
        "-y",
        "--yes",
        dest="confirm_mode",
        action="store_const",
        const="yes",
        help="Submit retries without prompting.",
    )
    check_parser.set_defaults(confirm_mode="prompt")

    retry_parser = subparsers.add_parser("retry", help="Submit only failed shards")
    add_shared_arguments(retry_parser)
    retry_parser.add_argument(
        "-y",
        "--yes",
        dest="confirm_mode",
        action="store_const",
        const="yes",
        help="Submit retries without prompting.",
    )
    retry_parser.set_defaults(confirm_mode="prompt")

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
    """Parse a comma-separated shard id list.

    Args:
        raw_value: Comma-separated shard ids, or None/empty for full array.
        num_shards: Optional upper bound used for range validation.

    Returns:
        Parsed shard ids.
    """
    if not raw_value:
        return []

    shard_ids: List[int] = []
    for raw_shard_id in raw_value.split(","):
        candidate = raw_shard_id.strip()
        if not candidate:
            continue

        if candidate.isdigit():
            shard_id = int(candidate)
        else:
            raise ValueError(f"Invalid shard id {candidate!r}; expected integers")

        if num_shards is not None and shard_id >= num_shards:
            raise ValueError(f"Invalid shard id {shard_id}; expected 0 <= shard_id < {num_shards}")

        shard_ids.append(shard_id)

    return shard_ids


def handle_run(args: argparse.Namespace, cfg: MMirageConfig, config_path: str) -> int:
    """Handle the canonical run command.

    Args:
        args: Parsed CLI namespace.
        cfg: Parsed MMIRAGE configuration object.
        config_path: Absolute path to the MMIRAGE YAML config file.

    Returns:
        Exit code for the run operation.
    """
    if args.shard_id is not None:
        return run_local(config_path, args.shard_id)
    return launch_pipeline(cfg, config_path, force_retry=args.force_retry)


def handle_submit(args: argparse.Namespace, cfg: MMirageConfig, config_path: str) -> int:
    """Submit a SLURM array job and optionally wait.

    Args:
        args: Parsed CLI namespace.
        cfg: Parsed MMIRAGE configuration object.
        config_path: Absolute path to the MMIRAGE YAML config file.

    Returns:
        Exit code for submission/wait outcome.
    """
    if require_slurm(cfg, "submit") != 0:
        return 1

    shard_ids = parse_shard_ids(args.shard_ids, cfg.loading_params.get_num_shards())
    job_id = submit_slurm_job(cfg, config_path, shard_ids)
    if job_id is None:
        return 1

    logger.info(f"Submitted SLURM job {job_id} for shard ids: {shard_ids or 'ALL'}")
    
    if not args.wait:
        return 0

    wait_for_slurm_job(job_id, cfg)
    failed_shards, summary = check_failed_shards(cfg)
    status_code = status_exit_code(failed_shards, summary)
    if status_code == 0:
        logger.info("All shards completed successfully")
    return status_code


def handle_check(args: argparse.Namespace, cfg: MMirageConfig, config_path: str) -> int:
    """Inspect shard status and optionally submit retries.

    Args:
        args: Parsed CLI namespace.
        cfg: Parsed MMIRAGE configuration object.
        config_path: Absolute path to the MMIRAGE YAML config file.

    Returns:
        Exit code based on shard status and optional retry submission.
    """
    failed_shards, summary = check_failed_shards(cfg)
    print(json.dumps(asdict(summary), indent=2))

    status_code = status_exit_code(failed_shards, summary)
    if not cfg.execution_params.is_slurm():
        return status_code

    if not args.retry:
        return status_code

    if not failed_shards:
        return status_code

    return submit_failed_shards(
        cfg=cfg,
        config_path=config_path,
        failed_shards=failed_shards,
        confirm_mode=args.confirm_mode,
    )


def handle_retry(args: argparse.Namespace, cfg: MMirageConfig, config_path: str) -> int:
    """Submit retries for failed shards only.

    Args:
        args: Parsed CLI namespace.
        cfg: Parsed MMIRAGE configuration object.
        config_path: Absolute path to the MMIRAGE YAML config file.

    Returns:
        Exit code for retry submission outcome.
    """
    if require_slurm(cfg, "retry") != 0:
        return 1

    failed_shards, summary = check_failed_shards(cfg)
    print(json.dumps(asdict(summary), indent=2))

    if not failed_shards:
        if summary.max_retries_exceeded > 0:
            logger.error("No retryable shards remain")
            return 1
        logger.info("All shards already succeeded.")
        return 0

    return submit_failed_shards(
        cfg=cfg,
        config_path=config_path,
        failed_shards=failed_shards,
        confirm_mode=args.confirm_mode,
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
        validate_edf_env_path(cfg)

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
