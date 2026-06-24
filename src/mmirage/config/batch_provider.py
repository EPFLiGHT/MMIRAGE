"""Provider-agnostic batch configuration contracts.

This module defines the shared configuration shape used by any future batch
submission provider (OpenAI, Anthropic, etc.).
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional


class OversizedRequestPolicy(str, Enum):
    """Policy for handling single requests that exceed the chunk byte limit."""

    ISOLATE = "isolate"
    REJECT = "reject"


@dataclass
class BatchRetryPolicy:
    """Retry behavior used by provider-neutral batch submission orchestration.

    Attributes:
        max_attempts: Maximum number of submission attempts for retryable errors.
        initial_backoff_seconds: Delay before the first retry attempt.
        backoff_multiplier: Multiplicative factor for subsequent retry delays.
    """

    max_attempts: int = 3
    initial_backoff_seconds: float = 2.0
    backoff_multiplier: float = 2.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.initial_backoff_seconds < 0:
            raise ValueError("initial_backoff_seconds must be >= 0")
        if self.backoff_multiplier < 1:
            raise ValueError("backoff_multiplier must be >= 1")


@dataclass
class BatchProviderConfig:
    """Shared contract for provider-specific batch configuration.

    Concrete provider configs should inherit from this dataclass and extend it
    with provider-specific settings. The fields here are intentionally provider
    neutral so chunking/submission orchestration can run through one typed path.

    Attributes:
        provider: Provider identifier (for example, "openai" or "anthropic").
        enabled: Whether batch submission mode is enabled.
        max_chunk_bytes: Maximum serialized request bytes per chunk.
            Defaults to 50 MB.
        max_requests_per_chunk: Optional hard cap on number of requests in a
            chunk. If None, no request-count cap is enforced.
        metadata_output_path: Base path where submission metadata receipts are saved.
            Submission writes suffixed files like ``.text.<run>.jsonl`` and
            ``.multimodal.<run>.jsonl`` from this base path.
        retry_policy: Retry policy used by the shared batch layer.
        oversized_request_policy: Handling policy when a single request exceeds
            ``max_chunk_bytes``. ``isolate`` creates a dedicated oversized
            chunk, while ``reject`` fails fast.
        extras: Provider-specific knobs that do not belong in the shared fields.
        credentials: Provider credentials required to submit chunks.
    """

    provider: str
    enabled: bool = True
    max_chunk_bytes: int = 50 * 1024 * 1024
    max_requests_per_chunk: Optional[int] = None
    metadata_output_path: str = ""
    retry_policy: BatchRetryPolicy = field(default_factory=BatchRetryPolicy)
    oversized_request_policy: OversizedRequestPolicy | str = OversizedRequestPolicy.ISOLATE
    extras: Dict[str, Any] = field(default_factory=dict)
    credentials: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.provider = self.provider.strip().lower()

        if not self.provider:
            raise ValueError("provider must be a non-empty string")
        if self.max_chunk_bytes < 1:
            raise ValueError("max_chunk_bytes must be >= 1")
        if self.max_requests_per_chunk is not None and self.max_requests_per_chunk < 1:
            raise ValueError("max_requests_per_chunk must be >= 1 when provided")
        if isinstance(self.oversized_request_policy, str):
            try:
                self.oversized_request_policy = OversizedRequestPolicy(
                    self.oversized_request_policy.strip().lower()
                )
            except ValueError as exc:
                raise ValueError(
                    "oversized_request_policy must be either 'isolate' or 'reject'"
                ) from exc
