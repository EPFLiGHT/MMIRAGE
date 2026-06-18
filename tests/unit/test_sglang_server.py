import os
import socket
import subprocess
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

_SRC = Path(__file__).resolve().parents[2] / "src"
_PKG = types.ModuleType("mmirage")
_PKG.__path__ = [str(_SRC / "mmirage")]
sys.modules.setdefault("mmirage", _PKG)

_CONFIG_MODULE = types.ModuleType(
    "mmirage.core.process.processors.image_gen.config"
)


@dataclass
class SGLangBackendConfig:
    api_key: str = "EMPTY"
    timeout_seconds: int = 900
    model_path: str = "Qwen/Qwen-Image"
    request_model: Optional[str] = None
    port: Optional[int] = None
    num_gpus: int = 1
    dtype: Optional[str] = None
    startup_timeout_seconds: int = 900
    extra_server_args: List[str] = field(default_factory=list)
    max_concurrent_requests: int = 1

    def resolved_base_url(self) -> str:
        if self.port is None:
            raise RuntimeError("SGLang server port has not been resolved yet.")
        return f"http://127.0.0.1:{self.port}/v1"


_CONFIG_MODULE.SGLangBackendConfig = SGLangBackendConfig
sys.modules.setdefault(
    "mmirage.core.process.processors.image_gen.config",
    _CONFIG_MODULE,
)

from mmirage.core.process.processors.image_gen import sglang_server


def _bind_localhost(port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", port))
    sock.listen(1)
    return sock


class StubProcess:
    pid = 12345

    def poll(self):
        return None


def test_next_available_port_skips_occupied_default_port():
    occupied = _bind_localhost(sglang_server.DEFAULT_SGLANG_PORT)
    try:
        config = SGLangBackendConfig(model_path="Qwen/Qwen-Image")
        port = sglang_server._next_available_port(config, set())
    finally:
        occupied.close()

    assert port == sglang_server.DEFAULT_SGLANG_PORT + 1


def test_shared_sglang_server_sets_internal_base_url_on_selected_port(monkeypatch):
    launched_ports = []

    def fake_launch(config):
        launched_ports.append(config.port)
        return StubProcess()

    def fake_wait(proc, config):
        assert config.port == sglang_server.DEFAULT_SGLANG_PORT

    stopped = []
    monkeypatch.setattr(sglang_server, "launch_sglang_server", fake_launch)
    monkeypatch.setattr(sglang_server, "wait_for_sglang_server", fake_wait)
    monkeypatch.setattr(sglang_server, "stop_sglang_server", lambda proc: stopped.append(proc))
    monkeypatch.delenv(sglang_server.MMIRAGE_SGLANG_BASE_URL, raising=False)

    config = SGLangBackendConfig(model_path="Qwen/Qwen-Image")
    with sglang_server.shared_sglang_server(config):
        assert launched_ports == [sglang_server.DEFAULT_SGLANG_PORT]
        assert os.environ[sglang_server.MMIRAGE_SGLANG_BASE_URL] == (
            f"http://127.0.0.1:{sglang_server.DEFAULT_SGLANG_PORT}/v1"
        )

    assert sglang_server.MMIRAGE_SGLANG_BASE_URL not in os.environ
    assert config.port is None
    assert len(stopped) == 1


def test_shared_sglang_server_retries_when_launch_reports_port_collision(monkeypatch):
    attempts = []

    def fake_launch(config):
        attempts.append(config.port)
        return StubProcess()

    def fake_wait(proc, config):
        if len(attempts) == 1:
            setattr(proc, "_mmirage_output_tail", ["OSError: address already in use"])
            raise RuntimeError("SGLang server exited before becoming ready")

    monkeypatch.setattr(sglang_server, "launch_sglang_server", fake_launch)
    monkeypatch.setattr(sglang_server, "wait_for_sglang_server", fake_wait)
    monkeypatch.setattr(sglang_server, "stop_sglang_server", lambda proc: None)
    monkeypatch.delenv(sglang_server.MMIRAGE_SGLANG_BASE_URL, raising=False)

    config = SGLangBackendConfig(model_path="Qwen/Qwen-Image")
    with sglang_server.shared_sglang_server(config):
        assert attempts == [
            sglang_server.DEFAULT_SGLANG_PORT,
            sglang_server.DEFAULT_SGLANG_PORT + 1,
        ]
        assert os.environ[sglang_server.MMIRAGE_SGLANG_BASE_URL].endswith(
            f":{sglang_server.DEFAULT_SGLANG_PORT + 1}/v1"
        )


def test_address_in_use_detection_does_not_retry_generic_readiness_errors():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(0.2)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        assert not sglang_server._is_address_in_use_failure(
            proc,
            sglang_server.DEFAULT_SGLANG_PORT,
            RuntimeError("readiness timed out"),
        )
    finally:
        proc.terminate()
        proc.wait(timeout=5)
