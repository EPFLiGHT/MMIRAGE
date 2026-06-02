"""SGLang Diffusion server image generation backend."""

from __future__ import annotations

import base64
import binascii
import io
import json
import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from PIL import Image as PILImage
except ImportError:  # pragma: no cover
    PILImage = None  # type: ignore


class SGLangImageBackend:
    """Image generation backend that calls a local SGLang Diffusion server.

    Supports two lifecycle modes controlled by ``SGLangBackendConfig.launch_mode``:

    - ``external``: connects to an already-running server.
    - ``managed``: spawns ``python -m sglang.launch_server`` as a subprocess,
      polls until the server is ready, and terminates it on :meth:`shutdown`.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "EMPTY",
        timeout_seconds: int = 900,
        request_model: Optional[str] = None,
        model_path: Optional[str] = None,
        validate_server: bool = True,
        max_concurrent_requests: int = 1,
        # managed-mode args
        _managed_process: Optional[subprocess.Popen] = None,
    ) -> None:
        """Initialize the SGLang HTTP backend.

        Args:
            base_url: Server base URL. Both of these are accepted:
                ``http://127.0.0.1:30010`` and
                ``http://127.0.0.1:30010/v1``.
            api_key: API key sent as ``Authorization: Bearer ...``.
                Use ``"EMPTY"`` for local unauthenticated servers.
            timeout_seconds: Per-request HTTP timeout.
            request_model: Optional model field to include in image requests.
                Most single-model SGLang deployments do not need this.
            model_path: Backward-compatible alias for ``request_model``.
                The actual local model path should normally be supplied when
                launching ``sglang serve``, not per request.
            validate_server: Whether to check server reachability at init time.
            max_concurrent_requests: Number of concurrent HTTP image requests
                issued from ``generate_batch``. Defaults to 1 for conservative
                server behavior.
        """
        if PILImage is None:  # pragma: no cover
            raise RuntimeError(
                "sglang backend requires Pillow. "
                "Install with: pip install -e .[image_gen]"
            )

        if not base_url:
            raise ValueError("SGLang base_url must be non-empty.")

        self._server_root_url, self._api_base_url = self._normalize_base_url(base_url)
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._request_model = request_model or model_path
        self._max_concurrent_requests = max(1, int(max_concurrent_requests))
        self._managed_process = _managed_process

        if validate_server:
            self._validate_server()

    # ------------------------------------------------------------------
    # Factory: managed server
    # ------------------------------------------------------------------

    @classmethod
    def from_managed_config(
        cls,
        model_path: str,
        port: int = 30010,
        num_gpus: int = 1,
        dtype: Optional[str] = None,
        api_key: str = "EMPTY",
        timeout_seconds: int = 900,
        startup_timeout_seconds: int = 120,
        extra_server_args: Optional[List[str]] = None,
        max_concurrent_requests: int = 1,
    ) -> SGLangImageBackend:
        """Launch a local SGLang Diffusion server and return a connected backend.

        The server is started as a subprocess.  This method blocks until the
        server responds to health-check requests or ``startup_timeout_seconds``
        elapses.

        Args:
            model_path: HuggingFace model ID or local model directory.
            port: TCP port the server should listen on.
            num_gpus: Tensor-parallelism degree (``--tp``).
            dtype: Optional model weight dtype (``--dtype float16`` etc.).
            api_key: API key used for subsequent requests.
            timeout_seconds: Per-request HTTP timeout for inference calls.
            startup_timeout_seconds: Seconds to wait for the server to become ready.
            extra_server_args: Extra CLI flags appended to the launch command.
            max_concurrent_requests: Concurrent HTTP image requests.
        """
        base_url = f"http://127.0.0.1:{port}/v1"
        proc = cls._launch_managed_server(
            model_path=model_path,
            port=port,
            num_gpus=num_gpus,
            dtype=dtype,
            extra_args=extra_server_args or [],
        )
        cls._wait_for_server(base_url, api_key, startup_timeout_seconds, proc)
        return cls(
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            request_model=model_path,
            validate_server=False,  # already confirmed ready
            max_concurrent_requests=max_concurrent_requests,
            _managed_process=proc,
        )

    @staticmethod
    def _launch_managed_server(
        model_path: str,
        port: int,
        num_gpus: int,
        dtype: Optional[str],
        extra_args: List[str],
    ) -> subprocess.Popen:
        """Spawn ``python -m sglang.launch_server`` and return the process handle."""
        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_path,
            "--port",
            str(port),
            "--tp",
            str(num_gpus),
        ]
        if dtype:
            cmd += ["--dtype", dtype]
        cmd += extra_args

        logger.info("Starting managed SGLang server: %s", " ".join(cmd))

        # Inherit the current environment so HF_HOME, CUDA_VISIBLE_DEVICES, etc.
        # are passed through to the server process.
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        logger.info("SGLang server process started (pid=%d)", proc.pid)
        return proc

    @staticmethod
    def _wait_for_server(
        base_url: str,
        api_key: str,
        startup_timeout_seconds: int,
        proc: subprocess.Popen,
    ) -> None:
        """Poll the server until it responds or the timeout elapses."""
        server_root = base_url.rstrip("/").removesuffix("/v1")
        health_urls = [
            f"{server_root}/health",
            f"{base_url}/models",
        ]
        deadline = time.monotonic() + startup_timeout_seconds
        poll_interval = 2.0

        logger.info(
            "Waiting up to %ds for SGLang server to become ready \u2026",
            startup_timeout_seconds,
        )

        while time.monotonic() < deadline:
            # Abort early if the process already exited.
            ret = proc.poll()
            if ret is not None:
                output = ""
                if proc.stdout:
                    try:
                        output = proc.stdout.read().decode(errors="replace")[-2000:]
                    except Exception:
                        pass
                raise RuntimeError(
                    f"SGLang server process exited unexpectedly with code {ret}.\n"
                    f"Last output:\n{output}"
                )

            for url in health_urls:
                try:
                    req = urllib.request.Request(
                        url,
                        headers={"Authorization": f"Bearer {api_key}"},
                    )
                    with urllib.request.urlopen(req, timeout=5):
                        logger.info("SGLang server is ready at %s", base_url)
                        return
                except Exception:
                    pass

            time.sleep(poll_interval)

        raise RuntimeError(
            f"SGLang server did not become ready within {startup_timeout_seconds}s. "
            "Check server logs or increase startup_timeout_seconds."
        )

    # ------------------------------------------------------------------
    # URL / HTTP helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_base_url(base_url: str) -> tuple[str, str]:
        """Return ``(server_root_url, api_base_url)`` from a user URL."""
        normalized = base_url.rstrip("/")

        if normalized.endswith("/v1"):
            server_root_url = normalized[: -len("/v1")].rstrip("/")
            api_base_url = normalized
        else:
            server_root_url = normalized
            api_base_url = f"{normalized}/v1"

        return server_root_url, api_base_url

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

    def _read_json(self, url: str, *, data: Optional[bytes] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Issue an HTTP request and parse the response as JSON."""
        req = urllib.request.Request(
            url,
            data=data,
            headers=self._headers(),
            method="POST" if data is not None else "GET",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout or self._timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode(errors="replace")
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
                f"SGLang server returned unexpected JSON response from {url}: {parsed!r}"
            )

        return parsed

    # ------------------------------------------------------------------
    # Server connectivity
    # ------------------------------------------------------------------

    def _validate_server(self) -> None:
        """Confirm the SGLang server is reachable before processing starts.

        SGLang Diffusion commonly exposes ``GET /models`` at the server root.
        Some OpenAI-compatible deployments expose ``GET /v1/models`` instead,
        so we try both to make local setups less brittle.
        """
        candidate_urls = [
            f"{self._server_root_url}/models",
            f"{self._api_base_url}/models",
        ]

        errors: List[str] = []
        for url in candidate_urls:
            try:
                self._read_json(url, timeout=10)
                logger.info("SGLang Diffusion server is reachable at %s", url)
                return
            except Exception as exc:
                errors.append(f"{url}: {exc}")

        raise RuntimeError(
            "Cannot reach SGLang Diffusion server. Ensure the server is running "
            "before starting the pipeline with launch_mode='external'. Tried:\n"
            + "\n".join(f"  - {err}" for err in errors)
        )

    # ------------------------------------------------------------------
    # Payload / response handling
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        prompt: str,
        negative_prompt: Optional[str],
        params: Dict[str, Any],
        seed: Optional[int],
    ) -> Dict[str, Any]:
        """Build an OpenAI-compatible image generation payload."""
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "response_format": "b64_json",
            "n": 1,
        }

        if self._request_model:
            payload["model"] = self._request_model

        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

        if seed is not None:
            payload["seed"] = int(seed)

        # OpenAI-compatible SGLang image API expects "size": "WIDTHxHEIGHT".
        if "size" in params:
            payload["size"] = params["size"]
        else:
            width = params.get("width")
            height = params.get("height")
            if width is not None and height is not None:
                payload["size"] = f"{int(width)}x{int(height)}"

        # Common diffusion-specific knobs. These are accepted by many SGLang
        # diffusion deployments, but any unsupported key will be rejected by the
        # server with a clear HTTP error.
        for key in ("num_inference_steps", "guidance_scale"):
            if key in params and params[key] is not None:
                payload[key] = params[key]

        # Forward extra model/pipeline-specific parameters, excluding fields
        # that were normalized above.
        reserved = {
            "width",
            "height",
            "size",
            "num_inference_steps",
            "guidance_scale",
            "generator",  # Diffusers-only; should never be sent to SGLang.
        }
        for key, value in params.items():
            if key not in reserved and value is not None:
                payload[key] = value

        return payload

    @staticmethod
    def _prompt_preview(prompt: str, limit: int = 80) -> str:
        compact = prompt.replace("\n", " ").strip()
        return compact if len(compact) <= limit else compact[: limit - 3] + "..."

    def _decode_image_response(self, result: Dict[str, Any], prompt: str) -> Any:
        """Decode the first ``b64_json`` image into a PIL Image."""
        try:
            data = result["data"]
            if not isinstance(data, list) or not data:
                raise KeyError("data[0]")

            b64_data = data[0]["b64_json"]
            if not isinstance(b64_data, str):
                raise TypeError("b64_json is not a string")
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "Unexpected SGLang image response for prompt "
                f"{self._prompt_preview(prompt)!r}: {result!r}"
            ) from exc

        try:
            img_bytes = base64.b64decode(b64_data)
        except (binascii.Error, ValueError) as exc:
            raise RuntimeError(
                "SGLang returned invalid base64 image data for prompt "
                f"{self._prompt_preview(prompt)!r}"
            ) from exc

        try:
            with PILImage.open(io.BytesIO(img_bytes)) as img:
                return img.convert("RGB")
        except Exception as exc:
            raise RuntimeError(
                "Could not decode SGLang image response for prompt "
                f"{self._prompt_preview(prompt)!r}"
            ) from exc

    # ------------------------------------------------------------------
    # Single-sample API call
    # ------------------------------------------------------------------

    def _call_api(
        self,
        prompt: str,
        negative_prompt: Optional[str],
        params: Dict[str, Any],
        seed: Optional[int],
    ) -> Any:
        """Call ``/v1/images/generations`` and return a PIL Image."""
        payload = self._build_payload(prompt, negative_prompt, params, seed)
        body = json.dumps(payload).encode("utf-8")

        url = f"{self._api_base_url}/images/generations"
        result = self._read_json(url, data=body, timeout=self._timeout)
        return self._decode_image_response(result, prompt)

    # ------------------------------------------------------------------
    # Backend interface
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        prompts: List[str],
        negative_prompts: Optional[List[Optional[str]]],
        params: Dict[str, Any],
        seeds: List[Optional[int]],
    ) -> List[Any]:
        """Generate images through SGLang.

        The OpenAI-compatible image endpoint is logically per prompt. This
        method preserves the backend interface by issuing one request per prompt.
        Requests are sequential by default. Set ``max_concurrent_requests > 1``
        only after confirming that the local SGLang server handles concurrent
        image generation reliably.
        """
        if negative_prompts is not None and len(negative_prompts) != len(prompts):
            raise ValueError(
                f"Expected {len(prompts)} negative prompts, got {len(negative_prompts)}"
            )

        if seeds and len(seeds) != len(prompts):
            raise ValueError(f"Expected {len(prompts)} seeds, got {len(seeds)}")

        def generate_one(i: int) -> Any:
            negative_prompt = negative_prompts[i] if negative_prompts is not None else None
            seed = seeds[i] if seeds else None
            return self._call_api(prompts[i], negative_prompt, params, seed)

        if self._max_concurrent_requests == 1 or len(prompts) <= 1:
            return [generate_one(i) for i in range(len(prompts))]

        with ThreadPoolExecutor(max_workers=self._max_concurrent_requests) as pool:
            futures = [pool.submit(generate_one, i) for i in range(len(prompts))]
            return [future.result() for future in futures]

    def shutdown(self) -> None:
        """Shut down the backend.  Terminates the managed server process if running."""
        if self._managed_process is not None:
            proc = self._managed_process
            self._managed_process = None
            logger.info("Stopping managed SGLang server (pid=%d) …", proc.pid)
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "SGLang server (pid=%d) did not terminate within 30 s; killing.",
                    proc.pid,
                )
                proc.kill()
                proc.wait()
            logger.info("SGLang server stopped.")