"""Provider-agnostic batch processing contracts and registry."""

from mmirage.core.process.batch.adapter import BatchSubmissionAdapter, BatchSubmissionResult
from mmirage.core.process.batch.collector import collect_and_merge
from mmirage.core.process.batch.chunking import BatchRequestChunker, RequestChunk
from mmirage.core.process.batch.openai_adapter import OpenAIBatchAdapter
from mmirage.core.process.batch.orchestrator import BatchSubmissionOrchestrator
from mmirage.core.process.batch.registry import BatchAdapterFactory, BatchAdapterRegistry
from mmirage.core.process.batch.status_checker import (
    extract_unique_provider_batches,
    run_status_checker,
)
from mmirage.config.openai_batch import OpenAIBatchConfig

__all__ = [
    "BatchSubmissionAdapter",
    "BatchSubmissionResult",
    "collect_and_merge",
    "BatchRequestChunker",
    "RequestChunk",
    "BatchSubmissionOrchestrator",
    "OpenAIBatchAdapter",
    "OpenAIBatchConfig",
    "BatchAdapterFactory",
    "BatchAdapterRegistry",
    "extract_unique_provider_batches",
    "run_status_checker",
]
