import json
from dataclasses import dataclass
from pathlib import Path

from mmirage.config.batch_provider import BatchProviderConfig
from mmirage.core.process.batch.adapter import BatchSubmissionAdapter
from mmirage.core.process.processors.batch_api.config import BatchApiProcessorConfig


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

        return BatchSubmissionResult(
            provider_batch_id=str(raw_result["id"]),
            status=str(raw_result["status"]),
            raw_response=raw_result,
        )

    def check_batch_status(self, provider_batch_id, config):
        from mmirage.core.process.batch.adapter import BatchSubmissionResult

        return BatchSubmissionResult(
            provider_batch_id=provider_batch_id, status="submitted", raw_response={}
        )

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


def test_batch_api_processor_exports_to_single_file_with_batch_ids(
    tmp_path, monkeypatch
):
    from mmirage.core.process.base import ProcessorRegistry
    from mmirage.core.process.batch.provider_resolution import (
        BatchProviderConfigRegistry,
    )
    from mmirage.core.process.batch.registry import BatchAdapterRegistry

    # Register provider config and adapter
    BatchProviderConfigRegistry.register("unit", UnitBatchConfig)
    BatchAdapterRegistry.register("unit", RecordingAdapter)

    # Create processor config and instantiate the batch API processor with export dir
    config = BatchApiProcessorConfig(
        type="batch_api",
        provider_config=UnitBatchConfig(
            provider="unit",
            max_chunk_bytes=10,
            metadata_output_path=str(tmp_path / "meta.jsonl"),
        ),
    )

    processor_cls = ProcessorRegistry.get_processor("batch_api")
    export_file = tmp_path / "exports" / "prompts.jsonl"
    processor = processor_cls(config, export_prompts_dir=str(export_file))

    assert processor._text_orchestrator is not None
    assert processor._multimodal_orchestrator is not None

    text_path = processor._text_orchestrator._export_prompts_path
    multi_path = processor._multimodal_orchestrator._export_prompts_path
    assert text_path is not None and multi_path is not None
    assert text_path.startswith(str(export_file).removesuffix(".jsonl") + ".")
    assert multi_path == text_path
    export_file = Path(text_path)

    # Submit one chunk to each orchestrator
    processor._text_orchestrator.add_requests(
        requests=[
            {"custom_id": "t1", "size_bytes": 6},
            {"custom_id": "t2", "size_bytes": 6},
        ],
        source_indices=[0, 1],
    )

    processor._multimodal_orchestrator.add_requests(
        requests=[
            {"custom_id": "m1", "size_bytes": 6},
            {"custom_id": "m2", "size_bytes": 6},
        ],
        source_indices=[2, 3],
    )

    assert export_file.exists()

    lines = [
        json.loads(line)
        for line in export_file.read_text(encoding="utf-8").splitlines()
    ]
    assert len(lines) == 2
    assert {line["request"]["custom_id"] for line in lines} == {"t1", "m1"}
    assert {line["batch_id"] for line in lines} == {
        "text-chunk-000001",
        "multimodal-chunk-000001",
    }
