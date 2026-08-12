"""Anthropic-specific batch configuration."""

from dataclasses import dataclass, field
from typing import Dict, Optional

from mmirage.config.batch_provider import BatchProviderConfig


@dataclass
class AnthropicBatchConfig(BatchProviderConfig):
    """Anthropic Messages Batches API configuration.

    Attributes:
        provider: Fixed provider identifier for Anthropic.
        model: Model name used in each Messages request body.
        max_tokens: Max tokens for the generated response.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        base_url: Optional base URL for API-compatible gateways.
        timeout_seconds: Request timeout in seconds.
        metadata: Metadata sent on batch creation.
    """

    provider: str = "anthropic"
    model: str = "claude-haiku-4-5"
    max_tokens: int = 8192
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    base_url: Optional[str] = None
    timeout_seconds: Optional[float] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()

        if not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.temperature is not None and not (0.0 <= self.temperature <= 1.0):
            raise ValueError("temperature must be in the range [0, 1] for Anthropic")
        if self.top_p is not None and not (0 < self.top_p <= 1):
            raise ValueError("top_p must be in the range (0, 1] when provided")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0 when provided")

        # Either temperature or top_p can be set or both can be None, but not both can be set, raise error if both are set
        if self.temperature is not None and self.top_p is not None:
            raise ValueError(
                "Both temperature and top_p are set in batch configuration, temperature and top_p are mutually exclusive parameters. Please set only one of them."
            )
        if self.temperature is None and self.top_p is None:
            self.temperature = 0.0
