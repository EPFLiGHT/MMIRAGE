from dataclasses import dataclass
import json

import pytest

from mmirage.config.batch_provider import BatchProviderConfig
from mmirage.core.process.batch.adapter import BatchSubmissionAdapter, BatchSubmissionResult
from mmirage.core.process.batch.provider_resolution import BatchProviderConfigRegistry
from mmirage.core.process.batch.registry import BatchAdapterRegistry
from mmirage.core.process.base import ProcessorRegistry
from mmirage.core.process.processors.llm.config import SGLangLLMConfig, SGLangServerArgs


class RecordingAdapter(BatchSubmissionAdapter):
    def __init__(self) -> None:
        self.submissions = []

    def build_request(self, custom_id, payload, config):
        return {"custom_id": custom_id, **dict(payload)}

    def estimate_request_bytes(self, request):
        return int(request["size_bytes"])

    def submit_chunk(self, chunk_id, requests, config):
        self.submissions.append(
            {
                "chunk_id": chunk_id,
                "requests": list(requests),
            }
        )
        return {"id": f"batch-{chunk_id}", "status": "submitted"}

    def parse_submission_result(self, raw_result):
        return BatchSubmissionResult(
            provider_batch_id=str(raw_result["id"]),
            status=str(raw_result["status"]),
            raw_response=raw_result,
        )

    def check_batch_status(self, provider_batch_id, config):
        return BatchSubmissionResult(
            provider_batch_id=provider_batch_id,
            status="submitted",
            raw_response={"id": provider_batch_id, "status": "submitted"},
        )

    def retrieve_results(self, provider_batch_id, config):
        return []


@pytest.fixture(autouse=True)
def clear_batch_registries():
    BatchProviderConfigRegistry.clear()
    BatchAdapterRegistry.clear()
    yield
    BatchProviderConfigRegistry.clear()
    BatchAdapterRegistry.clear()


def test_orchestrator_buffers_across_iterations_and_avoids_tiny_midstream_flush(tmp_path):
    from mmirage.core.process.batch.orchestrator import BatchSubmissionOrchestrator

    adapter = RecordingAdapter()
    config = BatchProviderConfig(
        provider="unit",
        max_chunk_bytes=10,
        metadata_output_path=str(tmp_path / "metadata.jsonl"),
    )
    orchestrator = BatchSubmissionOrchestrator(adapter=adapter, config=config)

    # Iteration 1: only 9 bytes total, should remain buffered and submit nothing.
    r1 = [{"custom_id": "a", "size_bytes": 6}, {"custom_id": "b", "size_bytes": 3}]
    out1 = orchestrator.add_requests(r1, [10, 11], {"phase": "iter1"})
    assert out1 == []
    assert len(adapter.submissions) == 0

    # Iteration 2: appending 2 bytes should emit one full chunk [6,3] and keep [2].
    r2 = [{"custom_id": "c", "size_bytes": 2}]
    out2 = orchestrator.add_requests(r2, [12], {"phase": "iter2"})
    assert len(out2) == 1
    assert len(adapter.submissions) == 1
    assert [x["size_bytes"] for x in adapter.submissions[0]["requests"]] == [6, 3]
    assert orchestrator.pending_count == 1

    # Finalize: emits the remaining tiny tail exactly once.
    out3 = orchestrator.finalize({"phase": "finalize"})
    assert len(out3) == 1
    assert len(adapter.submissions) == 2
    assert [x["size_bytes"] for x in adapter.submissions[1]["requests"]] == [2]
    assert orchestrator.pending_count == 0


def test_orchestrator_writes_provider_neutral_metadata_with_flush_reason(tmp_path):
    from mmirage.core.process.batch.orchestrator import BatchSubmissionOrchestrator

    metadata_path = tmp_path / "batch_metadata.jsonl"
    adapter = RecordingAdapter()
    config = BatchProviderConfig(
        provider="unit",
        max_chunk_bytes=10,
        metadata_output_path=str(metadata_path),
    )
    orchestrator = BatchSubmissionOrchestrator(adapter=adapter, config=config)

    orchestrator.add_requests(
        requests=[
            {"custom_id": "x1", "size_bytes": 8},
            {"custom_id": "x2", "size_bytes": 8},
        ],
        source_indices=[0, 1],
        model_params_snapshot={"model": "unit-model"},
    )
    orchestrator.finalize({"model": "unit-model"})

    lines = metadata_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    second = json.loads(lines[1])

    assert first["provider"] == "unit"
    assert first["flush_reason"] == "full_chunk"
    assert first["custom_id_to_source_index"] == {"x1": 0}
    assert isinstance(first["request_hash"], str) and len(first["request_hash"]) == 64

    assert second["flush_reason"] == "finalize"
    assert second["custom_id_to_source_index"] == {"x2": 1}
    assert second["provider_batch_id"].startswith("batch-chunk-")


@dataclass
class UnitBatchConfig(BatchProviderConfig):
    provider: str = "unit"
    unit_setting: str = "default"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.unit_setting.strip():
            raise ValueError("unit_setting must be a non-empty string")


def test_llm_processor_initializes_with_custom_provider(tmp_path):
    BatchProviderConfigRegistry.register("unit", UnitBatchConfig)
    BatchAdapterRegistry.register("unit", RecordingAdapter)

    config = SGLangLLMConfig(
        type="llm",
        server_args=SGLangServerArgs(model_path="dummy-model"),
        batch_provider=UnitBatchConfig(
            provider="unit",
            unit_setting="custom",
            metadata_output_path=str(tmp_path / "metadata.jsonl"),
        ),
    )

    processor_cls = ProcessorRegistry.get_processor("llm")
    processor = processor_cls(config)

    assert processor.batch_mode_enabled is True
    assert isinstance(processor._batch_provider_config, UnitBatchConfig)
    assert processor._batch_provider_config.provider == "unit"
    assert processor._batch_provider_config.unit_setting == "custom"
    assert isinstance(processor._batch_adapter, RecordingAdapter)


def test_llm_processor_skips_batch_setup_when_disabled(monkeypatch):
    class FakeEngine:
        def __init__(self, **_kwargs):
            return None

        def generate(self, **_kwargs):
            raise AssertionError("Synchronous generation should not run in this test")

        def shutdown(self):
            return None

    class FakeTokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return ""

    monkeypatch.setattr(
        "mmirage.core.process.processors.llm.llm_processor.sgl.Engine",
        FakeEngine,
    )
    monkeypatch.setattr(
        "mmirage.core.process.processors.llm.llm_processor.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )

    config = SGLangLLMConfig(
        type="llm",
        server_args=SGLangServerArgs(model_path="dummy-model"),
        batch_provider=BatchProviderConfig(provider="openai", enabled=False),
    )

    processor_cls = ProcessorRegistry.get_processor("llm")
    processor = processor_cls(config)

    assert processor.batch_mode_enabled is False
    assert processor._batch_adapter is None
    assert processor._batch_provider_config is None


def test_llm_processor_uses_sync_runtime_when_batch_provider_omitted(monkeypatch):
    class FakeEngine:
        def __init__(self, **_kwargs):
            return None

        def generate(self, **_kwargs):
            raise AssertionError("Synchronous generation should not run in this test")

        def shutdown(self):
            return None

    class FakeTokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return ""

    monkeypatch.setattr(
        "mmirage.core.process.processors.llm.llm_processor.sgl.Engine",
        FakeEngine,
    )
    monkeypatch.setattr(
        "mmirage.core.process.processors.llm.llm_processor.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )

    config = SGLangLLMConfig(
        type="llm",
        server_args=SGLangServerArgs(model_path="dummy-model"),
    )

    processor_cls = ProcessorRegistry.get_processor("llm")
    processor = processor_cls(config)

    assert processor.batch_mode_enabled is False
    assert isinstance(processor.llm, FakeEngine)
    assert isinstance(processor.tokenizer, FakeTokenizer)
    assert processor._batch_adapter is None
    assert processor._batch_provider_config is None
