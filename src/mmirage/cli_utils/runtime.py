"""Runtime/path helpers for the MMIRAGE CLI."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Sequence

from mmirage.config.config import MMirageConfig


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


def add_file_logging(log_file: str, level: str) -> None:
    """Add a file handler so logs are also written to disk."""
    expanded_log_file = os.path.abspath(os.path.expanduser(os.path.expandvars(log_file)))
    try:
        create_directories([str(Path(expanded_log_file).parent)])
    except OSError as exc:
        logger.warning("Unable to create log directory for %s: %s", expanded_log_file, exc)
        return

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and os.path.abspath(handler.baseFilename) == expanded_log_file:
            return

    try:
        file_handler = logging.FileHandler(expanded_log_file, mode="a", encoding="utf-8")
    except OSError as exc:
        logger.warning("Unable to open log file %s: %s", expanded_log_file, exc)
        return
    file_handler.setLevel(getattr(logging, level, logging.INFO))
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root_logger.addHandler(file_handler)


def setup_runtime(cfg: MMirageConfig, log_level: str) -> None:
    """Initialize runtime-level logging."""
    project_root = get_project_root(cfg)
    report_dir = expand_path(cfg.execution_params.report_dir, project_root)
    global_log_file = os.path.join(report_dir, f"{cfg.execution_params.job_name}.out")
    add_file_logging(global_log_file, log_level)
    logger.info("Writing logs to %s", global_log_file)
