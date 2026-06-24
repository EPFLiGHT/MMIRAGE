import pytest

from mmirage.config.batch_provider import BatchProviderConfig
from mmirage.core.process.batch.adapter import BatchSubmissionAdapter, BatchSubmissionResult


class SizeAwareTestAdapter(BatchSubmissionAdapter):
    def __init__(self) -> None:
        self.estimate_calls = []

    @property
    def adapter_name(self) -> str:
        return "size-aware-test-adapter"

    @property
    def adapter_version(self) -> str:
        return "1.0.0"

    def build_request(self, custom_id, payload, config):
        return {"custom_id": custom_id, **dict(payload)}

    def estimate_request_bytes(self, request):
        size = int(request["size_bytes"])
        self.estimate_calls.append(size)
        return size

    def submit_chunk(self, chunk_id, requests, config):
        return {"id": chunk_id, "status": "submitted"}

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


def _sizes_from_chunks(chunks):
    return [[request["size_bytes"] for request in chunk.requests] for chunk in chunks]


def test_chunker_splits_when_byte_limit_is_reached():
    from mmirage.core.process.batch.chunking import BatchRequestChunker

    adapter = SizeAwareTestAdapter()
    config = BatchProviderConfig(provider="unit", max_chunk_bytes=10)
    requests = [
        {"custom_id": "r1", "size_bytes": 4},
        {"custom_id": "r2", "size_bytes": 4},
        {"custom_id": "r3", "size_bytes": 4},
    ]

    chunks = BatchRequestChunker(adapter, config).chunk_requests(requests)

    assert _sizes_from_chunks(chunks) == [[4, 4], [4]]
    assert [chunk.total_bytes for chunk in chunks] == [8, 4]
    assert adapter.estimate_calls == [4, 4, 4]


def test_chunker_splits_when_max_requests_per_chunk_is_reached():
    from mmirage.core.process.batch.chunking import BatchRequestChunker

    adapter = SizeAwareTestAdapter()
    config = BatchProviderConfig(
        provider="unit",
        max_chunk_bytes=10_000,
        max_requests_per_chunk=2,
    )
    requests = [
        {"custom_id": "r1", "size_bytes": 1},
        {"custom_id": "r2", "size_bytes": 1},
        {"custom_id": "r3", "size_bytes": 1},
        {"custom_id": "r4", "size_bytes": 1},
        {"custom_id": "r5", "size_bytes": 1},
    ]

    chunks = BatchRequestChunker(adapter, config).chunk_requests(requests)

    assert _sizes_from_chunks(chunks) == [[1, 1], [1, 1], [1]]
    assert [chunk.total_requests for chunk in chunks] == [2, 2, 1]


def test_chunker_honors_exact_byte_boundary_without_flushing_early():
    from mmirage.core.process.batch.chunking import BatchRequestChunker

    adapter = SizeAwareTestAdapter()
    config = BatchProviderConfig(provider="unit", max_chunk_bytes=10)
    requests = [
        {"custom_id": "r1", "size_bytes": 6},
        {"custom_id": "r2", "size_bytes": 4},
        {"custom_id": "r3", "size_bytes": 1},
    ]

    chunks = BatchRequestChunker(adapter, config).chunk_requests(requests)

    assert _sizes_from_chunks(chunks) == [[6, 4], [1]]
    assert [chunk.total_bytes for chunk in chunks] == [10, 1]


def test_chunker_isolates_oversized_single_request_by_default(caplog):
    from mmirage.core.process.batch.chunking import BatchRequestChunker

    adapter = SizeAwareTestAdapter()
    config = BatchProviderConfig(provider="unit", max_chunk_bytes=10)
    requests = [
        {"custom_id": "r1", "size_bytes": 3},
        {"custom_id": "r2", "size_bytes": 25},
        {"custom_id": "r3", "size_bytes": 3},
    ]

    chunks = BatchRequestChunker(adapter, config).chunk_requests(requests)

    assert _sizes_from_chunks(chunks) == [[3], [25], [3]]
    assert [chunk.has_oversized_request for chunk in chunks] == [False, True, False]
    assert "oversized request" in caplog.text.lower()


def test_chunker_rejects_oversized_single_request_when_policy_is_reject():
    from mmirage.core.process.batch.chunking import BatchRequestChunker

    adapter = SizeAwareTestAdapter()
    config = BatchProviderConfig(
        provider="unit",
        max_chunk_bytes=10,
        oversized_request_policy="reject",
    )
    requests = [{"custom_id": "r1", "size_bytes": 11}]

    with pytest.raises(ValueError, match="oversized request"):
        BatchRequestChunker(adapter, config).chunk_requests(requests)
