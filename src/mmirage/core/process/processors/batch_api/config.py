"""Configuration for the batch API processor in MMIRAGE."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from mmirage.config.batch_provider import BatchProviderConfig
from mmirage.core.process.base import BaseProcessorConfig, ProcessorRegistry
from mmirage.core.process.batch.provider_resolution import (
    resolve_single_provider_config,
)
from mmirage.core.process.processors.llm.config import LLMOutputVar


@dataclass
class BatchApiProcessorConfig(BaseProcessorConfig):
    """Configuration for the batch API processor.

    Provider settings are declared inline in YAML and resolved to the matching
    provider config class::

        processors:
          - type: batch_api
            provider: openai
            model: gpt-4o-mini

    Attributes:
        provider_config: Resolved provider-specific batch configuration.
    """

    provider_config: Optional[BatchProviderConfig] = None

    @classmethod
    def from_raw(cls, data: Dict[str, Any]) -> "BatchApiProcessorConfig":
        """Build the config from a raw YAML block, dispatching on ``provider``."""
        block = {key: value for key, value in data.items() if key != "type"}
        return cls(
            type=data["type"], provider_config=resolve_single_provider_config(block)
        )


ProcessorRegistry.register_types("batch_api", BatchApiProcessorConfig, LLMOutputVar)
