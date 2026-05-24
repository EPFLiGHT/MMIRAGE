from dataclasses import dataclass
import json
import os

from mmirage.core.process.base import ProcessorRegistry

from mmirage.core.process.processors.llm.config import LLMProcessorConfig
from mmirage.core.process.batch.adapter import BatchSubmissionAdapter
from mmirage.config.batch_provider import BatchProviderConfig


class RecordingAdapter(BatchSubmissionAdapter):
    def __init__(self) -> None:
        self.submissions = []

    def build_request(self, custom_id, payload, config):
        return {"custom_id": custom_id, **dict(payload)}

    def estimate_request_bytes(self, request):
        return int(request.get("size_bytes", 0))

    def submit_chunk(self, chunk_id, requests, config):
        self.submissions.append({"chunk_id": chunk_id, "requests": list(requests)})
        return {"id": f"batch-{chunk_id}", "status": "submitted"}

    def parse_submission_result(self, raw_result):
        from mmirage.core.process.batch.adapter import BatchSubmissionResult

        return BatchSubmissionResult(provider_batch_id=str(raw_result["id"]), status=str(raw_result["status"]), raw_response=raw_result)

    def check_batch_status(self, provider_batch_id, config):
        from mmirage.core.process.batch.adapter import BatchSubmissionResult

        return BatchSubmissionResult(provider_batch_id=provider_batch_id, status="submitted", raw_response={})

    def retrieve_results(self, provider_batch_id, config):
        return []


@dataclass
class UnitBatchConfig(BatchProviderConfig):
    provider: str = "unit"
    unit_setting: str = "default"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.unit_setting.strip():
            raise ValueError("unit_setting must be a non-empty string")


def test_llm_processor_exports_to_text_and_multimodal_subdirs(tmp_path, monkeypatch):
    from mmirage.core.process.batch.registry import BatchAdapterRegistry
    from mmirage.core.process.batch.provider_resolution import BatchProviderConfigRegistry
    from mmirage.core.process.base import ProcessorRegistry

    # Register provider config and adapter
    BatchProviderConfigRegistry.register("unit", UnitBatchConfig)
    BatchAdapterRegistry.register("unit", RecordingAdapter)

    # Create processor config and instantiate LLM processor with export dir
    config = LLMProcessorConfig(
        type="llm",
        execution_mode="batch",
        batch=UnitBatchConfig(provider="unit", max_chunk_bytes=10, metadata_output_path=str(tmp_path / "meta.jsonl")),
    )

    processor_cls = ProcessorRegistry.get_processor("llm")
    export_root = tmp_path / "exports"
    processor = processor_cls(config, export_prompts_dir=str(export_root))

    assert processor._text_orchestrator is not None
    assert processor._multimodal_orchestrator is not None

    text_dir = processor._text_orchestrator._export_prompts_dir
    multi_dir = processor._multimodal_orchestrator._export_prompts_dir
    assert text_dir is not None and multi_dir is not None
    assert text_dir != multi_dir

    # Submit one chunk to each orchestrator
    processor._text_orchestrator.add_requests(
        requests=[{"custom_id": "t1", "size_bytes": 6}, {"custom_id": "t2", "size_bytes": 6}],
        source_indices=[0, 1],
    )

    processor._multimodal_orchestrator.add_requests(
        requests=[{"custom_id": "m1", "size_bytes": 6}, {"custom_id": "m2", "size_bytes": 6}],
        source_indices=[2, 3],
    )

    # Verify export files exist in both subdirectories
    assert os.path.isdir(text_dir)
    assert os.path.isdir(multi_dir)

    text_files = [p for p in os.listdir(text_dir) if p.startswith("batch_")]
    multi_files = [p for p in os.listdir(multi_dir) if p.startswith("batch_")]

    assert len(text_files) >= 1
    assert len(multi_files) >= 1

    # Verify contents
    text_content = (os.path.join(text_dir, text_files[0]))
    lines = [json.loads(l) for l in open(text_content, encoding="utf-8").read().splitlines()]
    assert any(line.get("custom_id") == "t1" for line in lines)

    multi_content = (os.path.join(multi_dir, multi_files[0]))
    lines = [json.loads(l) for l in open(multi_content, encoding="utf-8").read().splitlines()]
    assert any(line.get("custom_id") == "m1" for line in lines)
