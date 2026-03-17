"""Command-line interface for MMIRAGE pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from mmirage.config.config import MMirageConfig
from mmirage.config.utils import load_mmirage_config


logger = logging.getLogger(__name__)


def expand_path(path: str, project_root: Optional[str] = None) -> str:
    """Expand environment variables, user home and relative paths."""
    expanded = os.path.expanduser(os.path.expandvars(path))
    if not os.path.isabs(expanded) and project_root:
        expanded = os.path.join(project_root, expanded)
    return os.path.abspath(expanded)


def get_project_root(cfg: MMirageConfig) -> str:
    """Return the configured project root, or the current working directory."""
    project_root = cfg.execution_params.project_root
    if project_root:
        return expand_path(project_root)
    return os.getcwd()


def create_directories(paths: Sequence[str]) -> None:
    """Create directories if they do not already exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def validate_paths(cfg: MMirageConfig) -> None:
    """Validate pre-existing execution paths."""
    project_root = get_project_root(cfg)
    if cfg.execution_params.edf_env:
        edf_env = expand_path(cfg.execution_params.edf_env, project_root)
        if not os.path.exists(edf_env):
            raise FileNotFoundError(f"EDF environment file not found: {edf_env}")


def get_shard_state_dir(state_root: str, shard_id: int) -> str:
    """Return the state directory for a shard."""
    return os.path.join(state_root, f"shard_{shard_id}")


def get_shard_status(state_dir: str) -> Tuple[str, int]:
    """Read the current status and retry count for a shard."""
    status_file = os.path.join(state_dir, "status.json")
    if not os.path.exists(status_file):
        return ("missing", 0)

    try:
        with open(status_file, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read shard status from %s: %s", status_file, exc)
        return ("unknown", 0)

    return (str(data.get("status", "unknown")), int(data.get("retry_count", 0)))


def check_failed_shards(cfg: MMirageConfig) -> Tuple[List[int], dict]:
    """Return retryable failed shards and a compact summary."""
    state_root = cfg.loading_params.get_state_root()
    if not state_root:
        raise ValueError("loading_params.state_dir is required to check shard status")

    num_shards = cfg.loading_params.get_num_shards()
    max_retries = cfg.execution_params.max_retries
    failed_shards: List[int] = []
    success_count = 0
    exhausted_count = 0

    for shard_id in range(num_shards):
        status, retry_count = get_shard_status(get_shard_state_dir(state_root, shard_id))
        if status == "success":
            success_count += 1
            continue

        if retry_count >= max_retries:
            exhausted_count += 1
            logger.warning(
                "Shard %s exceeded retry budget (%s/%s)",
                shard_id,
                retry_count,
                max_retries,
            )
            continue

        failed_shards.append(shard_id)

    summary = {
        "total": num_shards,
        "successful": success_count,
        "failed": len(failed_shards),
        "max_retries_exceeded": exhausted_count,
    }
    return failed_shards, summary


def run_local(config_path: str, shard_id: Optional[int] = None) -> int:
    """Run one shard in the current Python environment."""
    command = [sys.executable, "-m", "mmirage.shard_process", "--config", config_path]
    env = os.environ.copy()
    if shard_id is not None:
        env["SLURM_ARRAY_TASK_ID"] = str(shard_id)

    logger.info("Running local shard processing: %s", " ".join(command))
    result = subprocess.run(command, env=env, check=False)
    return result.returncode


def build_sbatch_script(cfg: MMirageConfig, config_path: str) -> str:
    """Build the sbatch payload executed for each array task."""
    project_root = get_project_root(cfg)
    hf_home = expand_path(cfg.execution_params.hf_home, project_root)
    state_root = expand_path(cfg.loading_params.get_state_root(), project_root)

    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        f"export HF_HOME={shlex.quote(hf_home)}",
        f"export MMIRAGE_CONFIG={shlex.quote(config_path)}",
        f"mkdir -p {shlex.quote(hf_home)}",
        f"mkdir -p {shlex.quote(state_root)}",
        "srun_args=(--cpus-per-task ${SLURM_CPUS_PER_TASK:-1} --wait 60)",
    ]

    if cfg.execution_params.edf_env:
        edf_env = expand_path(cfg.execution_params.edf_env, project_root)
        lines.append(f"srun_args+=(--environment={shlex.quote(edf_env)})")

    lines.extend(
        [
            f"srun \"${{srun_args[@]}}\" {shlex.quote(sys.executable)} -m mmirage.shard_process --config \"$MMIRAGE_CONFIG\"",
            "echo \"Shard ${SLURM_ARRAY_TASK_ID:-0} completed\"",
        ]
    )
    return "\n".join(lines) + "\n"


def submit_slurm_job(
    cfg: MMirageConfig,
    config_path: str,
    shard_ids: Optional[Sequence[int]] = None,
) -> Optional[int]:
    """Submit a SLURM array job and return its job ID."""
    project_root = get_project_root(cfg)
    report_dir = expand_path(cfg.execution_params.report_dir, project_root)
    create_directories([report_dir])

    command = [
        "sbatch",
        "--parsable",
        f"--job-name={cfg.execution_params.job_name}",
        f"--chdir={project_root}",
        f"--output={os.path.join(report_dir, 'R-%x.%A_%a.out')}",
        f"--error={os.path.join(report_dir, 'R-%x.%A_%a.err')}",
        f"--nodes={cfg.execution_params.nodes}",
        f"--ntasks-per-node={cfg.execution_params.ntasks_per_node}",
        f"--gres=gpu:{cfg.execution_params.gpus}",
        f"--cpus-per-task={cfg.execution_params.cpus_per_task}",
        f"--time={cfg.execution_params.time_limit}",
        f"--account={cfg.execution_params.account}",
    ]

    if cfg.execution_params.reservation:
        command.append(f"--reservation={cfg.execution_params.reservation}")

    requested_shards = list(shard_ids or [])
    if requested_shards:
        command.append(f"--array={','.join(str(shard_id) for shard_id in requested_shards)}")
    else:
        num_shards = cfg.loading_params.get_num_shards()
        command.append(f"--array=0-{num_shards - 1}")

    logger.info("Submitting SLURM job: %s", " ".join(command))
    result = subprocess.run(
        command,
        input=build_sbatch_script(cfg, config_path),
        text=True,
        capture_output=True,
        check=False,
    )

    if result.returncode != 0:
        logger.error("sbatch failed: %s", result.stderr.strip())
        return None

    raw_job_id = result.stdout.strip().split(";", 1)[0]
    try:
        return int(raw_job_id)
    except ValueError:
        logger.error("Unable to parse job id from sbatch output: %s", result.stdout.strip())
        return None


def wait_for_slurm_job(job_id: int, cfg: MMirageConfig) -> None:
    """Wait for a SLURM job array to leave the queue."""
    logger.info("Waiting for SLURM job %s", job_id)
    while True:
        result = subprocess.run(
            ["squeue", "-h", "-j", str(job_id)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and not result.stdout.strip():
            break
        time.sleep(cfg.execution_params.poll_interval_seconds)

    if cfg.execution_params.settle_time_seconds > 0:
        logger.info("Waiting %ss for state files to settle", cfg.execution_params.settle_time_seconds)
        time.sleep(cfg.execution_params.settle_time_seconds)


def launch_pipeline(cfg: MMirageConfig, config_path: str, force_retry: bool = False) -> int:
    """Launch the pipeline according to execution mode and retry settings."""
    if not cfg.execution_params.is_slurm():
        return run_local(config_path, cfg.loading_params.get_shard_id())

    auto_retry = force_retry or cfg.execution_params.retry
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

        if not failed_shards and summary["max_retries_exceeded"] == 0:
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

    process_parser = subparsers.add_parser("process", help="Run a shard locally")
    add_shared_arguments(process_parser)
    process_parser.add_argument("--shard-id", type=int, default=None, help="Shard id override")

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

    launch_parser = subparsers.add_parser(
        "launch",
        help="(Deprecated) Use 'run'. Launch according to execution_params.mode and execution_params.retry",
    )
    add_shared_arguments(launch_parser)
    launch_parser.add_argument(
        "--force-retry",
        action="store_true",
        help="Enable retry orchestration even if execution_params.retry is false",
    )

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

    return parser


def _maybe_submit_retry_job(
    *,
    cfg: MMirageConfig,
    config_path: str,
    failed_shards: Sequence[int],
    interactive: bool,
) -> int:
    if not cfg.execution_params.is_slurm():
        logger.error("Retry submission requires execution_params.mode=slurm")
        return 1

    if not failed_shards:
        return 0

    if interactive:
        if not sys.stdin.isatty():
            logger.error("Non-interactive input detected; re-run with --no-interactive to auto-submit retries")
            return 1
        response = input(f"Retry {len(failed_shards)} shard(s)? (y/N) ")
        if response.strip().lower() != "y":
            print("Cancelled.")
            return 1

    job_id = submit_slurm_job(cfg, config_path, failed_shards)
    if job_id is None:
        return 1
    print(job_id)
    return 0


def parse_shard_ids(raw_value: Optional[str]) -> List[int]:
    """Parse a comma-separated shard id list."""
    if not raw_value:
        return []
    return [int(value.strip()) for value in raw_value.split(",") if value.strip()]


def main() -> None:
    """CLI entry point."""
    parser = build_argparser()
    args = parser.parse_args()
    configure_logging(args.log_level)

    try:
        config_path = os.path.abspath(args.config)
        cfg = load_mmirage_config(config_path)
        validate_paths(cfg)

        if args.command == "process":
            sys.exit(run_local(config_path, args.shard_id))

        if args.command == "submit":
            if not cfg.execution_params.is_slurm():
                logger.error("submit requires execution_params.mode=slurm")
                sys.exit(1)

            job_id = submit_slurm_job(cfg, config_path, parse_shard_ids(args.shard_ids))
            if job_id is None:
                sys.exit(1)

            print(job_id)
            if args.wait:
                wait_for_slurm_job(job_id, cfg)
                failed_shards, summary = check_failed_shards(cfg)
                sys.exit(0 if not failed_shards and summary["max_retries_exceeded"] == 0 else 1)
            sys.exit(0)

        if args.command == "check":
            failed_shards, summary = check_failed_shards(cfg)
            print(json.dumps(summary, indent=2))

            if not cfg.execution_params.is_slurm():
                sys.exit(0 if not failed_shards and summary["max_retries_exceeded"] == 0 else 1)

            if args.summary_only or not args.retry:
                sys.exit(0 if not failed_shards and summary["max_retries_exceeded"] == 0 else 1)

            if not failed_shards:
                sys.exit(0 if summary["max_retries_exceeded"] == 0 else 1)

            sys.exit(
                _maybe_submit_retry_job(
                    cfg=cfg,
                    config_path=config_path,
                    failed_shards=failed_shards,
                    interactive=bool(args.interactive),
                )
            )

        if args.command == "retry":
            if not cfg.execution_params.is_slurm():
                logger.error("retry requires execution_params.mode=slurm")
                sys.exit(1)

            failed_shards, summary = check_failed_shards(cfg)
            print(json.dumps(summary, indent=2))

            if not failed_shards:
                if summary["max_retries_exceeded"] > 0:
                    logger.error("No retryable shards remain")
                    sys.exit(1)
                print("All shards already succeeded.")
                sys.exit(0)

            if args.interactive:
                response = input(f"Retry {len(failed_shards)} shard(s)? (y/N) ")
                if response.strip().lower() != "y":
                    print("Cancelled.")
                    sys.exit(1)

            job_id = submit_slurm_job(cfg, config_path, failed_shards)
            if job_id is None:
                sys.exit(1)

            print(job_id)
            sys.exit(0)

        if args.command in {"launch", "run"}:
            if args.command == "launch":
                logger.warning("'launch' is deprecated; use 'run' instead")
            sys.exit(launch_pipeline(cfg, config_path, force_retry=args.force_retry))

    except Exception as exc:
        logger.error("Error: %s", exc, exc_info=logger.isEnabledFor(logging.DEBUG))
        sys.exit(1)


if __name__ == "__main__":
    main()
