"""SGLang Diffusion image generation backend.

This backend talks to a local SGLang Diffusion HTTP server using the
OpenAI-compatible image generation endpoint:

    POST /v1/images/generations

It supports two usage patterns:

1. External mode:
   The caller starts SGLang separately and passes its base URL.

2. Managed mode:
   This class starts `sglang serve` as a subprocess, waits until it is ready,
   and stops it during shutdown.
"""

from __future__ import annotations

import base64
import binascii
import io
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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from PIL import Image as PILImage
except ImportError:  # pragma: no cover
    PILImage = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class ManagedSGLangConfig:
    """Configuration for a managed SGLang Diffusion server."""

    model_path: str
    port: int = 30010
    num_gpus: int = 1
    server_command: Optional[str] = None
    backend: Optional[str] = None
    api_key: str = "EMPTY"
    request_model: Optional[str] = None
    timeout_seconds: int = 900
    startup_timeout_seconds: int = 300
    max_concurrent_requests: int = 1
    extra_server_args: Sequence[str] = field(default_factory=tuple)
    env: Optional[Mapping[str, str]] = None


class SGLangImageBackend:
    """Image backend for SGLang Diffusion's OpenAI-compatible image API."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "EMPTY",
        timeout_seconds: int = 900,
        request_model: Optional[str] = None,
        validate_server: bool = True,
        max_concurrent_requests: int = 1,
        managed_process: Optional[subprocess.Popen[bytes]] = None,
    ) -> None:
        if PILImage is None:  # pragma: no cover
            raise RuntimeError(
                "SGLangImageBackend requires Pillow. Install it with `pip install Pillow`."
            )
        if not base_url or not base_url.strip():
            raise ValueError("base_url must be non-empty")

        self._server_root_url, self._api_base_url = self._normalize_base_url(base_url)
        self._api_key = api_key
        self._timeout_seconds = int(timeout_seconds)
        self._request_model = request_model
        self._max_concurrent_requests = max(1, int(max_concurrent_requests))
        self._managed_process = managed_process

        if validate_server:
            self.validate_server()

    # ---------------------------------------------------------------------
    # Construction
    # ---------------------------------------------------------------------

    @classmethod
    def from_managed_config(cls, config: ManagedSGLangConfig) -> "SGLangImageBackend":
        """Start `sglang serve`, wait for readiness, and return a backend."""
        base_url = f"http://127.0.0.1:{config.port}/v1"
        proc = cls._start_managed_server(config)

        try:
            cls._wait_for_server(
                base_url=base_url,
                api_key=config.api_key,
                timeout_seconds=config.startup_timeout_seconds,
                proc=proc,
            )
        except Exception:
            cls._terminate_process(proc, grace_seconds=30)
            raise

        return cls(
            base_url=base_url,
            api_key=config.api_key,
            timeout_seconds=config.timeout_seconds,
            request_model=config.request_model,
            validate_server=False,
            max_concurrent_requests=config.max_concurrent_requests,
            managed_process=proc,
        )

    @staticmethod
    def _start_managed_server(config: ManagedSGLangConfig) -> subprocess.Popen[bytes]:
        if not config.model_path:
            raise ValueError("ManagedSGLangConfig.model_path must be non-empty")

        server_command = config.server_command or shutil.which("sglang")
        if server_command is None:
            raise RuntimeError(
                "Could not find the `sglang` executable. Install `sglang[diffusion]` "
                "in the active environment or pass ManagedSGLangConfig(server_command=...)."
            )

        cmd = [
            server_command,
            "serve",
            "--model-path",
            config.model_path,
            "--port",
            str(config.port),
            "--num-gpus",
            str(config.num_gpus),
        ]

        if config.backend:
            cmd += ["--backend", config.backend]

        cmd += list(config.extra_server_args)

        env = os.environ.copy()
        if config.env:
            env.update(dict(config.env))

        logger.info("Starting SGLang Diffusion server: %s", _shell_join(cmd))

        proc: subprocess.Popen[bytes] = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )

        threading.Thread(
            target=SGLangImageBackend._log_process_output,
            args=(proc,),
            daemon=True,
        ).start()

        logger.info("SGLang Diffusion server process started with pid=%d", proc.pid)
        return proc

    @staticmethod
    def _log_process_output(proc: subprocess.Popen[bytes]) -> None:
        if proc.stdout is None:
            return

        for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            if line:
                logger.info("[sglang-server] %s", line)

    @staticmethod
    def _wait_for_server(
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: int,
        proc: subprocess.Popen[bytes],
    ) -> None:
        server_root, api_base = SGLangImageBackend._normalize_base_url(base_url)
        candidate_urls = (
            f"{server_root}/models",
            f"{api_base}/models",
            f"{server_root}/health",
        )

        deadline = time.monotonic() + timeout_seconds
        last_error = "server did not respond"

        logger.info("Waiting up to %ds for SGLang server readiness", timeout_seconds)

        while time.monotonic() < deadline:
            retcode = proc.poll()
            if retcode is not None:
                raise RuntimeError(
                    f"SGLang server exited before becoming ready with code {retcode}. "
                    "Check the [sglang-server] logs above."
                )

            for url in candidate_urls:
                try:
                    SGLangImageBackend._read_json_static(
                        url=url,
                        api_key=api_key,
                        timeout_seconds=5,
                    )
                    logger.info("SGLang server is ready at %s", base_url)
                    return
                except Exception as exc:  # Keep polling; report the last failure on timeout.
                    last_error = f"{url}: {exc}"

            time.sleep(2.0)

        raise RuntimeError(
            f"SGLang server did not become ready within {timeout_seconds}s. "
            f"Last readiness error: {last_error}"
        )

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def validate_server(self) -> None:
        """Raise if the configured SGLang server cannot be reached."""
        candidate_urls = (
            f"{self._server_root_url}/models",
            f"{self._api_base_url}/models",
            f"{self._server_root_url}/health",
        )

        errors: List[str] = []
        for url in candidate_urls:
            try:
                self._read_json(url, timeout_seconds=10)
                logger.info("SGLang server is reachable at %s", url)
                return
            except Exception as exc:
                errors.append(f"{url}: {exc}")

        raise RuntimeError(
            "Cannot reach SGLang server. Tried:\n"
            + "\n".join(f"  - {error}" for error in errors)
        )

    def generate_batch(
        self,
        prompts: Sequence[str],
        negative_prompts: Optional[Sequence[Optional[str]]] = None,
        params: Optional[Mapping[str, Any]] = None,
        seeds: Optional[Sequence[Optional[int]]] = None,
    ) -> List[Any]:
        """Generate one image per prompt and return PIL Images."""
        params = params or {}
        prompts = list(prompts)

        if negative_prompts is not None and len(negative_prompts) != len(prompts):
            raise ValueError(
                f"Expected {len(prompts)} negative prompts, got {len(negative_prompts)}"
            )
        if seeds is not None and len(seeds) != len(prompts):
            raise ValueError(f"Expected {len(prompts)} seeds, got {len(seeds)}")

        def generate_one(index: int) -> Any:
            negative_prompt = (
                negative_prompts[index] if negative_prompts is not None else None
            )
            seed = seeds[index] if seeds is not None else None
            return self.generate_one(
                prompt=prompts[index],
                negative_prompt=negative_prompt,
                params=params,
                seed=seed,
            )

        if self._max_concurrent_requests == 1 or len(prompts) <= 1:
            return [generate_one(i) for i in range(len(prompts))]

        with ThreadPoolExecutor(max_workers=self._max_concurrent_requests) as pool:
            futures = [pool.submit(generate_one, i) for i in range(len(prompts))]
            return [future.result() for future in futures]

    def generate_one(
        self,
        *,
        prompt: str,
        negative_prompt: Optional[str] = None,
        params: Optional[Mapping[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> Any:
        """Generate a single image and return a PIL Image."""
        payload = self._build_payload(
            prompt=prompt,
            negative_prompt=negative_prompt,
            params=params or {},
            seed=seed,
        )
        result = self._read_json(
            f"{self._api_base_url}/images/generations",
            payload=payload,
            timeout_seconds=self._timeout_seconds,
        )
        return self._decode_image_response(result, prompt)

    def shutdown(self) -> None:
        """Stop the managed SGLang process, if this backend started one."""
        proc = self._managed_process
        self._managed_process = None

        if proc is None:
            return

        logger.info("Stopping managed SGLang server with pid=%d", proc.pid)
        self._terminate_process(proc, grace_seconds=30)
        logger.info("Managed SGLang server stopped")

    def __enter__(self) -> "SGLangImageBackend":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.shutdown()

    # ---------------------------------------------------------------------
    # Payload and response handling
    # ---------------------------------------------------------------------

    def _build_payload(
        self,
        *,
        prompt: str,
        negative_prompt: Optional[str],
        params: Mapping[str, Any],
        seed: Optional[int],
    ) -> JsonDict:
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be non-empty")

        payload: JsonDict = {
            "prompt": prompt,
            "n": int(params.get("n", 1)),
            "response_format": params.get("response_format", "b64_json"),
        }

        if payload["n"] != 1:
            raise ValueError(
                "This backend currently expects n=1 because it decodes only one image "
                "per request. Use multiple prompts for batch generation."
            )

        if self._request_model:
            payload["model"] = self._request_model
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        if seed is not None:
            payload["seed"] = int(seed)

        size = self._extract_size(params)
        if size:
            payload["size"] = size

        # Common diffusion knobs. Keep names exactly as the server expects.
        for key in (
            "num_inference_steps",
            "guidance_scale",
            "output-quality",
            "output-compression",
        ):
            if key in params and params[key] is not None:
                payload[key] = params[key]

        # Friendly aliases for callers that prefer Python identifiers.
        aliases = {
            "output_quality": "output-quality",
            "output_compression": "output-compression",
        }
        for source_key, target_key in aliases.items():
            if source_key in params and params[source_key] is not None:
                payload[target_key] = params[source_key]

        reserved = {
            "prompt",
            "negative_prompt",
            "model",
            "n",
            "response_format",
            "size",
            "width",
            "height",
            "seed",
            "generator",
            "num_inference_steps",
            "guidance_scale",
            "output-quality",
            "output-compression",
            "output_quality",
            "output_compression",
        }

        for key, value in params.items():
            if key not in reserved and value is not None:
                payload[key] = value

        return payload

    @staticmethod
    def _extract_size(params: Mapping[str, Any]) -> Optional[str]:
        if params.get("size"):
            return str(params["size"])

        width = params.get("width")
        height = params.get("height")
        if width is None or height is None:
            return None

        return f"{int(width)}x{int(height)}"

    @staticmethod
    def _decode_image_response(result: Mapping[str, Any], prompt: str) -> Any:
        try:
            data = result["data"]
            if not isinstance(data, list) or not data:
                raise TypeError("response field `data` must be a non-empty list")

            first = data[0]
            if not isinstance(first, Mapping):
                raise TypeError("response field `data[0]` must be an object")

            b64_data = first["b64_json"]
            if not isinstance(b64_data, str):
                raise TypeError("response field `data[0].b64_json` must be a string")
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                "Unexpected SGLang image response for prompt "
                f"{_prompt_preview(prompt)!r}: {result!r}"
            ) from exc

        try:
            image_bytes = base64.b64decode(b64_data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise RuntimeError(
                "SGLang returned invalid base64 image data for prompt "
                f"{_prompt_preview(prompt)!r}"
            ) from exc

        try:
            with PILImage.open(io.BytesIO(image_bytes)) as image:  # type: ignore[union-attr]
                return image.convert("RGB")
        except Exception as exc:
            raise RuntimeError(
                "Could not decode SGLang image response for prompt "
                f"{_prompt_preview(prompt)!r}"
            ) from exc

    # ---------------------------------------------------------------------
    # HTTP helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _normalize_base_url(base_url: str) -> Tuple[str, str]:
        normalized = base_url.strip().rstrip("/")

        if normalized.endswith("/v1"):
            api_base_url = normalized
            server_root_url = normalized[: -len("/v1")].rstrip("/")
        else:
            server_root_url = normalized
            api_base_url = f"{normalized}/v1"

        return server_root_url, api_base_url

    def _read_json(
        self,
        url: str,
        *,
        payload: Optional[Mapping[str, Any]] = None,
        timeout_seconds: Optional[int] = None,
    ) -> JsonDict:
        return self._read_json_static(
            url=url,
            api_key=self._api_key,
            payload=payload,
            timeout_seconds=timeout_seconds or self._timeout_seconds,
        )

    @staticmethod
    def _read_json_static(
        *,
        url: str,
        api_key: str,
        payload: Optional[Mapping[str, Any]] = None,
        timeout_seconds: int,
    ) -> JsonDict:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        }
        if body is not None:
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(
            url,
            data=body,
            headers=headers,
            method="POST" if body is not None else "GET",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"SGLang server returned HTTP {exc.code} for {url}: {body_text}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach SGLang server at {url}: {exc}") from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"SGLang server returned non-JSON response from {url}: {raw[:500]}"
            ) from exc

        if not isinstance(parsed, dict):
            raise RuntimeError(
                f"SGLang server returned unexpected JSON from {url}: {parsed!r}"
            )

        return parsed

    # ---------------------------------------------------------------------
    # Process helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _terminate_process(proc: subprocess.Popen[bytes], *, grace_seconds: int) -> None:
        if proc.poll() is not None:
            return

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


def _prompt_preview(prompt: str, limit: int = 80) -> str:
    compact = " ".join(prompt.split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def _shell_join(parts: Iterable[str]) -> str:
    """Small fallback for shlex.join on older environments."""
    try:
        import shlex

        return shlex.join(list(parts))
    except Exception:  # pragma: no cover
        return " ".join(parts)