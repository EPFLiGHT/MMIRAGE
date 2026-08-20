"""End-to-end tests for the CustomProcessor, running a real spawned worker pool.

what is tested : pool wiring, worker initialization, and pickling of rows and results across the process boundary

Note : Nothing is mocked here then it costs time for starting real workers
"""

import pytest

from mmirage.core.process.processors.custom.config import (
    CustomOutputVar,
    CustomProcessorConfig,
)
from mmirage.core.process.processors.custom.custom_processor import CustomProcessor
from mmirage.core.process.variables import VariableEnvironment


@pytest.fixture
def output_var() -> CustomOutputVar:
    return CustomOutputVar(name="result_var")


def build_processor(script_path, function_name: str, **overrides) -> CustomProcessor:
    """Build a processor backed by a real spawned pool."""
    return CustomProcessor(
        CustomProcessorConfig(
            type="custom",
            script_path=str(script_path),
            function_name=function_name,
            max_workers=2,
            timeout_ms=30_000,
            fallback_value="TEST_FALLBACK",
            **overrides,
        )
    )


def test_real_pool_executes_user_script(tmp_path, output_var):
    """Verify a batch round-trips through real workers, in order, with real results."""
    script = tmp_path / "dummy_script.py"
    script.write_text("def analyze_text(row): return f\"analyzed {row['index']}\"")

    processor = build_processor(script, "analyze_text")
    try:
        batch = [VariableEnvironment({"index": i}) for i in range(6)]
        processed = processor.batch_process_sample(batch, output_var)
    finally:
        processor.shutdown()

    assert [env.get("result_var") for env in processed] == [
        f"analyzed {i}" for i in range(6)
    ]


def test_real_pool_applies_fallback_on_user_exception(tmp_path, output_var):
    """Verify an exception raised inside a worker comes back as the fallback value."""
    script = tmp_path / "raising_script.py"
    script.write_text(
        "def analyze_text(row):\n"
        "    if row['index'] == 1:\n"
        "        raise ValueError('boom')\n"
        "    return 'ok'\n"
    )

    processor = build_processor(script, "analyze_text", max_errors=2)
    try:
        batch = [VariableEnvironment({"index": i}) for i in range(3)]
        processed = processor.batch_process_sample(batch, output_var)
    finally:
        processor.shutdown()

    assert [env.get("result_var") for env in processed] == [
        "ok",
        "TEST_FALLBACK",
        "ok",
    ]
