"""Single-node Slurm orchestration for shared SGLang image generation."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import List

from mmirage.cli_utils.runtime import expand_path, get_project_root
from mmirage.config.utils import load_mmirage_config
from mmirage.core.process.processors.image_gen.sglang_server import (
    get_sglang_server_config,
    shared_sglang_server,
)

logger = logging.getLogger(__name__)


def _parse_shard_ids(raw_value: str, num_shards: int) -> List[int]:
    if not raw_value:
        return list(range(num_shards))
    shard_ids = [int(value) for value in raw_value.split(",") if value]
    invalid = [shard_id for shard_id in shard_ids if not 0 <= shard_id < num_shards]
    if invalid:
        raise ValueError(
            f"Invalid shard ids {invalid}; expected 0 <= shard_id < {num_shards}"
        )
    return shard_ids


def main() -> None:
    """Launch one shared server and run all requested shard workers against it."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser("Run shared-SGLang MMIRAGE shard workers.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--shard-ids", default="")
    args = ap.parse_args()

    cfg = load_mmirage_config(args.config)
    sglang = get_sglang_server_config(cfg)
    if sglang is None:
        raise RuntimeError(
            "sglang_job requires an image_gen processor with backend='sglang'"
        )

    shard_ids = _parse_shard_ids(args.shard_ids, cfg.loading_params.get_num_shards())
    processes: List[subprocess.Popen[bytes]] = []
    report_dir = Path(
        expand_path(cfg.execution_params.report_dir, get_project_root(cfg))
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    log_prefix = f"R-{cfg.execution_params.job_name}.{job_id}"

    with shared_sglang_server(sglang), ExitStack() as stack:
        for shard_id in shard_ids:
            env = os.environ.copy()
            env["SLURM_ARRAY_TASK_ID"] = str(shard_id)
            command = [
                sys.executable,
                "-m",
                "mmirage.shard_process",
                "--config",
                args.config,
            ]
            stdout_path = report_dir / f"{log_prefix}_{shard_id}.out"
            stderr_path = report_dir / f"{log_prefix}_{shard_id}.err"
            stdout = stack.enter_context(stdout_path.open("w", encoding="utf-8"))
            stderr = stack.enter_context(stderr_path.open("w", encoding="utf-8"))
            logger.info(
                "Starting shard worker %d: %s (stdout=%s, stderr=%s)",
                shard_id,
                " ".join(command),
                stdout_path,
                stderr_path,
            )
            processes.append(
                subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr)
            )

        return_codes = [proc.wait() for proc in processes]

    failed = [
        shard_id
        for shard_id, return_code in zip(shard_ids, return_codes)
        if return_code != 0
    ]
    if failed:
        raise RuntimeError(f"Shard workers failed: {failed}")


if __name__ == "__main__":
    main()
