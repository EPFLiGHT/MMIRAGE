import pytest
from batch_fixtures import RecordingAdapter, UnitBatchConfig

from mmirage.core.process.batch.provider_resolution import BatchProviderConfigRegistry
from mmirage.core.process.batch.registry import BatchAdapterRegistry


@pytest.fixture(autouse=True)
def clear_batch_registries():
    """Keep a provider registered by one test out of the next one."""
    BatchProviderConfigRegistry.clear()
    BatchAdapterRegistry.clear()
    yield
    BatchProviderConfigRegistry.clear()
    BatchAdapterRegistry.clear()


@pytest.fixture
def unit_provider():
    BatchProviderConfigRegistry.register("unit", UnitBatchConfig)
    BatchAdapterRegistry.register("unit", RecordingAdapter)
