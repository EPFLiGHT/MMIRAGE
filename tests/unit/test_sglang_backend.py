"""Unit tests for the SGLang image generation backend and config."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


class TestSGLangBackendConfig(unittest.TestCase):
    """Tests for SGLangBackendConfig field defaults and validation."""

    def _make_config(self, **kwargs):
        from mmirage.core.process.processors.image_gen.config import SGLangBackendConfig

        return SGLangBackendConfig(**kwargs)

    def test_default_launch_mode_is_managed(self):
        cfg = self._make_config(model_path="Qwen/Qwen-Image")
        self.assertEqual(cfg.launch_mode, "managed")

    def test_default_server_env_is_empty_dict(self):
        cfg = self._make_config(model_path="Qwen/Qwen-Image")
        self.assertEqual(cfg.server_env, {})

    def test_default_max_concurrent_requests_is_one(self):
        cfg = self._make_config(model_path="Qwen/Qwen-Image")
        self.assertEqual(cfg.max_concurrent_requests, 1)

    def test_default_request_model_is_none(self):
        cfg = self._make_config(model_path="Qwen/Qwen-Image")
        self.assertIsNone(cfg.request_model)

    def test_server_env_round_trips(self):
        env = {"HF_HOME": "/some/path", "TRITON_CACHE_DIR": "/other"}
        cfg = self._make_config(model_path="Qwen/Qwen-Image", server_env=env)
        self.assertEqual(cfg.server_env, env)

    def test_max_concurrent_requests_zero_raises(self):
        with self.assertRaises(ValueError):
            self._make_config(model_path="Qwen/Qwen-Image", max_concurrent_requests=0)

    def test_max_concurrent_requests_negative_raises(self):
        with self.assertRaises(ValueError):
            self._make_config(model_path="Qwen/Qwen-Image", max_concurrent_requests=-1)

    def test_managed_mode_derives_base_url_from_port(self):
        cfg = self._make_config(model_path="Qwen/Qwen-Image", port=31000)
        self.assertEqual(cfg.base_url, "http://127.0.0.1:31000/v1")

    def test_external_mode_requires_no_model_path(self):
        cfg = self._make_config(
            launch_mode="external",
            base_url="http://127.0.0.1:30010/v1",
        )
        self.assertEqual(cfg.launch_mode, "external")

    def test_managed_mode_requires_model_path(self):
        from mmirage.core.process.processors.image_gen.config import SGLangBackendConfig

        with self.assertRaises(ValueError):
            SGLangBackendConfig(launch_mode="managed")  # no model_path

    def test_request_model_set(self):
        cfg = self._make_config(model_path="Qwen/Qwen-Image", request_model="MyModel")
        self.assertEqual(cfg.request_model, "MyModel")


class TestSGLangRequestPayload(unittest.TestCase):
    """Tests that _build_payload includes/omits 'model' correctly."""

    def _make_backend(self, request_model=None):
        from mmirage.core.process.processors.image_gen.backends.sglang_backend import (
            SGLangImageBackend,
        )

        return SGLangImageBackend(
            base_url="http://127.0.0.1:30010/v1",
            request_model=request_model,
            validate_server=False,
        )

    def test_no_model_in_payload_when_request_model_is_none(self):
        backend = self._make_backend(request_model=None)
        payload = backend._build_payload(
            prompt="a red sunset",
            negative_prompt=None,
            params={},
            seed=None,
        )
        self.assertNotIn("model", payload)

    def test_model_in_payload_when_request_model_is_set(self):
        backend = self._make_backend(request_model="Qwen/Qwen-Image")
        payload = backend._build_payload(
            prompt="a red sunset",
            negative_prompt=None,
            params={},
            seed=None,
        )
        self.assertIn("model", payload)
        self.assertEqual(payload["model"], "Qwen/Qwen-Image")


class TestManagedCommandConstruction(unittest.TestCase):
    """Tests that managed-mode builds the correct subprocess command."""

    def test_uses_sglang_serve_and_num_gpus(self):
        from mmirage.core.process.processors.image_gen.backends.sglang_backend import (
            ManagedSGLangConfig,
            SGLangImageBackend,
        )

        cfg = ManagedSGLangConfig(
            model_path="Qwen/Qwen-Image",
            port=30010,
            num_gpus=1,
        )

        captured_cmd = []

        def fake_popen(cmd, **kwargs):
            captured_cmd.extend(cmd)
            proc = MagicMock()
            proc.pid = 12345
            proc.stdout = None
            return proc

        with patch("shutil.which", return_value="/usr/bin/sglang"), patch(
            "subprocess.Popen", side_effect=fake_popen
        ):
            SGLangImageBackend._start_managed_server(cfg)

        # Must use 'sglang serve', not 'python -m sglang.launch_server'
        self.assertIn("sglang", captured_cmd[0])
        self.assertIn("serve", captured_cmd)
        self.assertIn("--model-path", captured_cmd)
        self.assertIn("--num-gpus", captured_cmd)

        # Must NOT use python -m sglang.launch_server or --tp
        self.assertNotIn("python", captured_cmd)
        # '-m' must not appear as a standalone argument (not inside --model-path)
        self.assertNotIn("-m", captured_cmd)
        self.assertNotIn("sglang.launch_server", captured_cmd)
        self.assertNotIn("--tp", captured_cmd)

    def test_raises_when_sglang_not_on_path(self):
        from mmirage.core.process.processors.image_gen.backends.sglang_backend import (
            ManagedSGLangConfig,
            SGLangImageBackend,
        )

        cfg = ManagedSGLangConfig(model_path="Qwen/Qwen-Image")
        with patch("shutil.which", return_value=None):
            with self.assertRaises(RuntimeError) as ctx:
                SGLangImageBackend._start_managed_server(cfg)
        self.assertIn("sglang", str(ctx.exception).lower())

    def test_server_env_is_merged_into_subprocess_env(self):
        from mmirage.core.process.processors.image_gen.backends.sglang_backend import (
            ManagedSGLangConfig,
            SGLangImageBackend,
        )

        cfg = ManagedSGLangConfig(
            model_path="Qwen/Qwen-Image",
            env={"HF_HOME": "/custom/hf"},
        )

        captured_env = {}

        def fake_popen(cmd, **kwargs):
            captured_env.update(kwargs.get("env", {}))
            proc = MagicMock()
            proc.pid = 1
            proc.stdout = None
            return proc

        with patch("shutil.which", return_value="/usr/bin/sglang"), patch(
            "subprocess.Popen", side_effect=fake_popen
        ):
            SGLangImageBackend._start_managed_server(cfg)

        self.assertEqual(captured_env.get("HF_HOME"), "/custom/hf")


class TestReadinessUrls(unittest.TestCase):
    """Tests that the readiness check polls /models first."""

    def test_wait_for_server_polls_models_endpoint(self):
        from mmirage.core.process.processors.image_gen.backends.sglang_backend import (
            SGLangImageBackend,
        )

        polled_urls = []

        def fake_read_json_static(*, url, api_key, timeout_seconds, **kw):
            polled_urls.append(url)
            if "/models" in url:
                return {"data": []}
            raise RuntimeError("not found")

        proc = MagicMock()
        proc.poll.return_value = None  # still running

        with patch.object(
            SGLangImageBackend, "_read_json_static", staticmethod(fake_read_json_static)
        ):
            SGLangImageBackend._wait_for_server(
                base_url="http://127.0.0.1:30010/v1",
                api_key="EMPTY",
                timeout_seconds=10,
                proc=proc,
            )

        # /models endpoint must be among those polled (and preferably first)
        models_urls = [u for u in polled_urls if u.endswith("/models")]
        self.assertTrue(
            len(models_urls) >= 1,
            f"Expected at least one /models URL, got: {polled_urls}",
        )
        # /models should be the first URL tried
        self.assertIn("/models", polled_urls[0])


if __name__ == "__main__":
    unittest.main()
