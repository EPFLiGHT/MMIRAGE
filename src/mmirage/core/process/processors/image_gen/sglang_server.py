"""Shared SGLang Diffusion server lifecycle for MMIRAGE orchestration."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import shutil
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from contextlib import contextmanager
from typing import Any, Deque, Iterator, List, Optional, Sequence

from mmirage.core.process.processors.image_gen.config import SGLangBackendConfig


logger = logging.getLogger(__name__)

MMIRAGE_SGLANG_BASE_URL = "MMIRAGE_SGLANG_BASE_URL"


def get_sglang_server_config(cfg: Any) -> Optional[SGLangBackendConfig]:
    """Return the shared server config when this run uses backend='sglang'."""
    configs = [
        processor.sglang
        for processor in cfg.processors
        if getattr(processor, "type", None) == "image_gen"
        and getattr(processor, "backend", None) == "sglang"
    ]
    if not configs:
        return None
    if len(configs) > 1:
        raise ValueError("Only one backend='sglang' image_gen processor is supported per run")
    return configs[0]


def launch_sglang_server(config: SGLangBackendConfig) -> subprocess.Popen[bytes]:
    """Launch the shared SGLang Diffusion server."""
    executable = shutil.which("sglang")
    if executable is None:
        raise RuntimeError(
            "Could not find the `sglang` executable. Activate or install an "
            "environment containing SGLang before running MMIRAGE."
        )
    _require_sglang_diffusion_installation()

    cmd = [
        executable,
        "serve",
        "--model-path",
        config.model_path,
        "--port",
        str(config.port),
        "--num-gpus",
        str(config.num_gpus),
    ]
    if config.dtype:
        cmd += ["--dtype", config.dtype]
    cmd += list(config.extra_server_args)

    logger.info("Starting shared SGLang Diffusion server: %s", _shell_join(cmd))
    proc: subprocess.Popen[bytes] = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    output_tail: Deque[str] = deque(maxlen=40)
    setattr(proc, "_mmirage_output_tail", output_tail)
    threading.Thread(
        target=_log_process_output,
        args=(proc, output_tail),
        daemon=True,
    ).start()
    logger.info("Shared SGLang Diffusion server started with pid=%d", proc.pid)
    return proc


def _require_sglang_diffusion_installation() -> None:
    """Raise an actionable error when the active SGLang install lacks diffusion."""
    try:
        spec = importlib.util.find_spec("sglang.multimodal_gen")
    except (AttributeError, ImportError, KeyError, ValueError) as exc:
        raise RuntimeError(
            "The active SGLang installation is incomplete or inconsistent: "
            "`sglang.multimodal_gen` could not be imported. Install the SGLang "
            "diffusion dependencies in the environment used by the MMIRAGE job "
            "with `pip install -e '.[image_gen]'`, or directly with "
            "`uv pip install 'sglang[diffusion]==0.5.10'`."
        ) from exc

    if spec is None:
        raise RuntimeError(
            "The active SGLang installation does not include "
            "`sglang.multimodal_gen`. Install the SGLang diffusion dependencies "
            "in the environment used by the MMIRAGE job with "
            "`pip install -e '.[image_gen]'`, or directly with "
            "`uv pip install 'sglang[diffusion]==0.5.10'`."
        )


def wait_for_sglang_server(
    proc: subprocess.Popen[bytes],
    config: SGLangBackendConfig,
) -> None:
    """Wait until the shared SGLang server reports readiness."""
    server_root = f"http://127.0.0.1:{config.port}"
    candidate_urls = (
        f"{server_root}/models",
        f"{server_root}/health",
        f"{server_root}/v1/models",
    )
    deadline = time.monotonic() + config.startup_timeout_seconds
    last_error = "server did not respond"

    logger.info(
        "Waiting up to %ds for shared SGLang server readiness",
        config.startup_timeout_seconds,
    )
    while time.monotonic() < deadline:
        retcode = proc.poll()
        if retcode is not None:
            raise RuntimeError(
                f"SGLang server exited before becoming ready with code {retcode}. "
                f"Recent SGLang output:\n{_format_output_tail(proc)}"
            )

        for url in candidate_urls:
            try:
                _read_json(url, config.api_key)
                logger.info("Shared SGLang server is ready at %s", url)
                return
            except Exception as exc:
                last_error = f"{url}: {exc}"
        time.sleep(2.0)

    raise RuntimeError(
        "SGLang server did not become ready within "
        f"{config.startup_timeout_seconds}s. Last readiness error: {last_error}\n"
        f"Recent SGLang output:\n{_format_output_tail(proc)}"
    )


def stop_sglang_server(proc: subprocess.Popen[bytes], grace_seconds: int = 30) -> None:
    """Stop the shared SGLang server process group."""
    if proc.poll() is not None:
        return

    logger.info("Stopping shared SGLang server with pid=%d", proc.pid)
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        logger.exception("Failed to terminate SGLang process group; trying proc.terminate()")
        proc.terminate()

    try:
        proc.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        logger.warning(
            "SGLang server pid=%d did not stop within %ds; killing it",
            proc.pid,
            grace_seconds,
        )

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except Exception:
        logger.exception("Failed to kill SGLang process group; trying proc.kill()")
        proc.kill()
    proc.wait()


@contextmanager
def shared_sglang_server(config: Optional[SGLangBackendConfig]) -> Iterator[None]:
    """Launch one server for an orchestration scope and publish its base URL."""
    if config is None:
        yield
        return

    proc = launch_sglang_server(config)
    previous_base_url = os.environ.get(MMIRAGE_SGLANG_BASE_URL)
    try:
        wait_for_sglang_server(proc, config)
        os.environ[MMIRAGE_SGLANG_BASE_URL] = config.resolved_base_url()
        yield
    finally:
        if previous_base_url is None:
            os.environ.pop(MMIRAGE_SGLANG_BASE_URL, None)
        else:
            os.environ[MMIRAGE_SGLANG_BASE_URL] = previous_base_url
        stop_sglang_server(proc)


def _log_process_output(proc: subprocess.Popen[bytes], output_tail: Deque[str]) -> None:
    if proc.stdout is None:
        return
    for raw_line in proc.stdout:
        line = raw_line.decode("utf-8", errors="replace").rstrip()
        if line:
            output_tail.append(line)
            logger.info("[sglang-server] %s", line)


def _format_output_tail(proc: subprocess.Popen[bytes]) -> str:
    output_tail: Sequence[str] = getattr(proc, "_mmirage_output_tail", ())
    return "\n".join(output_tail) or "(no server output captured)"


def _read_json(url: str, api_key: str) -> None:
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read readiness endpoint {url}: {exc}") from exc


def _shell_join(parts: List[str]) -> str:
    try:
        import shlex

        return shlex.join(parts)
    except Exception:  # pragma: no cover
        return " ".join(parts)
