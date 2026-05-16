"""Registry and factory for provider batch adapters."""

import os
from typing import Dict, Type

from mmirage.config.batch_provider import BatchProviderConfig
from mmirage.core.process.batch.adapter import BatchSubmissionAdapter


class BatchAdapterRegistry:
    """Provider-to-adapter registry with factory helpers.

    This class centralizes provider registration and fail-fast adapter
    instantiation with credential validation.
    """

    _registry: Dict[str, Type[BatchSubmissionAdapter]] = dict()
    _bootstrapped: bool = False

    @classmethod
    def _bootstrap_builtin_adapters(cls) -> None:
        if cls._bootstrapped:
            return

        # Local import avoids import cycles while ensuring built-ins are available.
        from mmirage.core.process.batch.openai_adapter import OpenAIBatchAdapter

        cls.register("openai", OpenAIBatchAdapter)
        cls._bootstrapped = True

    @classmethod
    def register(cls, provider: str, adapter_cls: Type[BatchSubmissionAdapter]) -> None:
        """Register an adapter class under a provider key."""
        provider_key = provider.strip().lower()
        if not provider_key:
            raise ValueError("provider must be a non-empty string")
        cls._registry[provider_key] = adapter_cls

    @classmethod
    def clear(cls) -> None:
        """Clear all registered adapters.

        Intended for tests and isolated bootstrapping logic.
        """
        cls._registry.clear()
        cls._bootstrapped = False

    @classmethod
    def resolve(cls, provider: str) -> Type[BatchSubmissionAdapter]:
        """Resolve a provider key to a registered adapter class."""
        cls._bootstrap_builtin_adapters()
        provider_key = provider.strip().lower()
        if provider_key not in cls._registry:
            raise ValueError(
                f"Unknown batch provider '{provider}'. "
                f"Available providers: {list(cls._registry.keys())}"
            )
        return cls._registry[provider_key]

    @classmethod
    def create(cls, config: BatchProviderConfig) -> BatchSubmissionAdapter:
        """Instantiate an adapter for a provider config with credential checks."""
        adapter_cls = cls.resolve(config.provider)

        missing_credentials = []
        for req_key in adapter_cls.required_credentials:
            credential_value = (config.credentials.get(req_key, "") or "").strip()
            if credential_value:
                continue

            env_var = f"{config.provider.upper()}_{req_key.upper()}"
            env_value = (os.environ.get(env_var, "") or "").strip()
            if env_value:
                config.credentials[req_key] = env_value
                continue

            missing_credentials.append(req_key)

        if missing_credentials:
            raise ValueError(
                f"Missing credentials for provider '{config.provider}': {missing_credentials}"
            )
        return adapter_cls()


class BatchAdapterFactory:
    """Compatibility alias around registry-based adapter creation."""

    @classmethod
    def from_config(cls, config: BatchProviderConfig) -> BatchSubmissionAdapter:
        """Create an adapter from provider config via registry resolution."""
        return BatchAdapterRegistry.create(config)
