import base64
import importlib
import json
import sys
from types import SimpleNamespace

import pytest

from mmirage.config.anthropic_batch import AnthropicBatchConfig
from mmirage.core.process.batch.provider_resolution import resolve_single_provider_config
from mmirage.core.process.batch.registry import BatchAdapterFactory


@pytest.fixture(autouse=True)
def anthropic_api_key(monkeypatch):
    """The API key is only ever read from the environment."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


def _load_anthropic_adapter(monkeypatch, fake_client_cls):
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=fake_client_cls))
    module = importlib.import_module("mmirage.core.process.batch.anthropic_adapter")
    importlib.reload(module)
    return module.AnthropicBatchAdapter


def test_anthropic_build_request_normalizes_messages_and_images(tmp_path, monkeypatch):
    image_bytes = b"\xff\xd8\xff\xe0testjpeg"
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(image_bytes)

    class FakeAnthropic:
        def __init__(self, **kwargs):
            pass

    AnthropicBatchAdapter = _load_anthropic_adapter(monkeypatch, FakeAnthropic)

    config = AnthropicBatchConfig(model="claude-haiku-4.5")
    adapter = AnthropicBatchAdapter()
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe image"},
                    {"type": "image_url", "image_url": {"url": str(image_path)}},
                ],
            }
        ]
    }

    request = adapter.build_request(custom_id="vision-1", payload=payload, config=config)

    assert request["custom_id"] == "vision-1"
    assert request["params"]["model"] == "claude-haiku-4.5"
    assert request["params"]["max_tokens"] == config.max_tokens

    content = request["params"]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "describe image"}
    assert content[1]["type"] == "image"

    encoded = content[1]["source"]["data"]
    assert base64.b64decode(encoded) == image_bytes


def test_anthropic_estimate_request_bytes_matches_utf8_json_size(monkeypatch):
    class FakeAnthropic:
        def __init__(self, **kwargs):
            pass

    AnthropicBatchAdapter = _load_anthropic_adapter(monkeypatch, FakeAnthropic)
    adapter = AnthropicBatchAdapter()

    request = {"custom_id": "accented", "params": {"message": "caf\u00e9"}}
    expected = len(
        json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )

    assert adapter.estimate_request_bytes(request) == expected


def test_anthropic_submit_chunk_uses_messages_batches(monkeypatch):
    captured = {}

    class FakeBatches:
        def create(self, **kwargs):
            captured["create_kwargs"] = kwargs
            return SimpleNamespace(id="batch_123", status="submitted")

    class FakeMessages:
        def __init__(self):
            self.batches = FakeBatches()

    class FakeAnthropic:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.messages = FakeMessages()

    AnthropicBatchAdapter = _load_anthropic_adapter(monkeypatch, FakeAnthropic)

    config = AnthropicBatchConfig()
    adapter = AnthropicBatchAdapter()
    requests = [
        adapter.build_request(
            custom_id="r1",
            payload={"messages": [{"role": "user", "content": "Hi"}]},
            config=config,
        )
    ]

    raw_result = adapter.submit_chunk(chunk_id="chunk-01", requests=requests, config=config)

    assert captured["client_kwargs"]["api_key"] == "test-key"
    assert captured["create_kwargs"]["requests"] == requests
    assert captured["create_kwargs"]["metadata"]["chunk_id"] == "chunk-01"
    assert raw_result["id"] == "batch_123"
    assert raw_result["status"] == "submitted"


def test_anthropic_check_batch_status_falls_back_to_env_api_key(monkeypatch):
    captured = {}

    class FakeBatches:
        def retrieve(self, provider_batch_id):
            return SimpleNamespace(id=provider_batch_id, status="completed")

    class FakeMessages:
        def __init__(self):
            self.batches = FakeBatches()

    class FakeAnthropic:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.messages = FakeMessages()

    AnthropicBatchAdapter = _load_anthropic_adapter(monkeypatch, FakeAnthropic)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-test-key")
    config = AnthropicBatchConfig()
    adapter = AnthropicBatchAdapter()

    result = adapter.check_batch_status(provider_batch_id="batch_env", config=config)

    assert captured["client_kwargs"]["api_key"] == "env-test-key"
    assert result.provider_batch_id == "batch_env"


def test_anthropic_retrieve_results_normalizes_custom_id_and_generated_text(monkeypatch):
    class FakeBatches:
        def retrieve(self, provider_batch_id):
            return SimpleNamespace(id=provider_batch_id, status="completed")

        def results(self, provider_batch_id):
            return [
                {
                    "request": {"custom_id": "c1"},
                    "result": {
                        "type": "succeeded",
                        "message": {
                            "content": [
                                {"type": "text", "text": "Hello"},
                                {"type": "text", "text": " world"},
                            ]
                        },
                    },
                }
            ]

    class FakeMessages:
        def __init__(self):
            self.batches = FakeBatches()

    class FakeAnthropic:
        def __init__(self, **kwargs):
            self.messages = FakeMessages()

    AnthropicBatchAdapter = _load_anthropic_adapter(monkeypatch, FakeAnthropic)

    config = AnthropicBatchConfig()
    adapter = AnthropicBatchAdapter()

    rows = adapter.retrieve_results(provider_batch_id="batch_1", config=config)

    assert rows == [
        {
            "request": {"custom_id": "c1"},
            "result": {
                "type": "succeeded",
                "message": {
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "text", "text": " world"},
                    ]
                },
            },
            "custom_id": "c1",
            "generated_text": "Hello world",
        }
    ]


def test_resolve_single_provider_config_accepts_anthropic():
    config = resolve_single_provider_config({"provider": "anthropic"})
    assert isinstance(config, AnthropicBatchConfig)


def test_factory_raises_when_anthropic_api_key_env_var_is_missing(monkeypatch):
    class FakeAnthropic:
        def __init__(self, **kwargs):
            pass

    _load_anthropic_adapter(monkeypatch, FakeAnthropic)

    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        BatchAdapterFactory.from_config(AnthropicBatchConfig())