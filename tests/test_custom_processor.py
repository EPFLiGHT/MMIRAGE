"""Tests for the CustomProcessor and its configuration/lifecycle."""

import concurrent.futures
import sys
from unittest.mock import MagicMock, patch

import pytest
from pebble import ProcessExpired

from mmirage.config.utils import load_mmirage_config
from mmirage.core.process.base import AutoProcessor
from mmirage.core.process.processors.custom import worker
from mmirage.core.process.processors.custom.config import (
    CustomOutputVar,
    CustomProcessorConfig,
)
from mmirage.core.process.processors.custom.custom_processor import CustomProcessor
from mmirage.core.process.variables import VariableEnvironment


# applies to every test in this file: no CustomProcessor ever starts real workers
@pytest.fixture(autouse=True)
def mock_pebble_pool():
    """Mock the pebble ProcessPool to prevent real multiprocessing overhead in tests.

    This patches the pool precisely at the module level where CustomProcessor imports it.
    """
    with patch(
        "mmirage.core.process.processors.custom.custom_processor.ProcessPool"
    ) as mock_pool:
        pool_instance = MagicMock()
        mock_pool.return_value = pool_instance
        yield pool_instance


@pytest.fixture
def dummy_script(tmp_path) -> str:
    """Creates a temporary dummy script file to bypass the os.path.exists check."""
    script_file = tmp_path / "dummy_script.py"
    script_file.write_text("def analyze_text(row): return 'analyzed'")
    return str(script_file)


@pytest.fixture
def clean_worker_state():
    """Reset the worker module globals, which persist in-process between tests."""
    original_path = list(sys.path)
    yield
    worker._USER_FUNCTION = None
    sys.path[:] = original_path


@pytest.fixture
def base_config(dummy_script) -> CustomProcessorConfig:
    """Provides a default CustomProcessorConfig for testing."""
    return CustomProcessorConfig(
        type="custom",
        script_path=dummy_script,
        function_name="analyze_text",
        max_workers=2,
        timeout_ms=1000,
        max_timeouts=2,
        max_errors=2,
        fallback_value="TEST_FALLBACK",
    )


def test_config_registration_and_loading(tmp_path, dummy_script):
    """Verify that YAML parser successfully resolves type 'custom' to CustomProcessorConfig."""
    yaml_content = f"""
    processors:
      - type: "custom"
        script_path: "{dummy_script}"
        function_name: "analyze_text"
        max_workers: 4
        timeout_ms: 2000
        max_timeouts: 5
        max_errors: 3
        fallback_value: "TIMEOUT_FALLBACK"

    loading_params:
      num_shards: 1
      batch_size: 10

    processing_params:
      inputs: []
      outputs:
        - name: "custom_result"
          type: "custom"
      output_schema: {{}}
    """
    conf_file = tmp_path / "config_test_custom.yaml"
    conf_file.write_text(yaml_content)

    # Load configuration
    config = load_mmirage_config(str(conf_file))

    # Assert Processor Config parsing
    proc_config = config.processors[0]
    assert isinstance(proc_config, CustomProcessorConfig)
    assert proc_config.script_path == dummy_script
    assert proc_config.max_workers == 4
    assert proc_config.max_timeouts == 5
    assert proc_config.max_errors == 3

    # Assert OutputVar config parsing
    out_var = config.processing_params.outputs[0]
    assert isinstance(out_var, CustomOutputVar)


def test_lazy_processor_lookup():
    """Verify the AutoProcessor resolves the string 'custom' to the CustomProcessor class."""
    processor_class = AutoProcessor.from_name("custom")
    assert processor_class is CustomProcessor


def test_custom_processor_strict_ordering(base_config, mock_pebble_pool):
    """Verify batch output order strictly matches input order despite out-of-order completion."""
    processor = CustomProcessor(base_config)
    out_var = CustomOutputVar(name="result_var")

    envs = [
        VariableEnvironment({"input": "row0"}),
        VariableEnvironment({"input": "row1"}),
        VariableEnvironment({"input": "row2"}),
    ]

    # Setup mock futures to return distinctive results
    mock_futures = [MagicMock(), MagicMock(), MagicMock()]
    mock_futures[0].result.return_value = "res_0"
    mock_futures[1].result.return_value = "res_1"
    mock_futures[2].result.return_value = "res_2"

    # Pool schedules them sequentially (0, 1, 2)
    mock_pebble_pool.schedule.side_effect = mock_futures

    # Simulate pebble workers finishing arbitrarily out of order (2 finishes first, then 0, then 1)
    def fake_as_completed(futures):
        yield mock_futures[2]
        yield mock_futures[0]
        yield mock_futures[1]

    with patch("concurrent.futures.as_completed", side_effect=fake_as_completed):
        processed_envs = processor.batch_process_sample(envs, out_var)

    # Resulting environments must map strictly to their original indices
    assert processed_envs[0].get("result_var") == "res_0"
    assert processed_envs[1].get("result_var") == "res_1"
    assert processed_envs[2].get("result_var") == "res_2"


def test_custom_processor_fallback_injection(base_config, mock_pebble_pool):
    """Verify that solitary timeouts and exceptions trigger the fallback value without crashing."""
    processor = CustomProcessor(base_config)
    out_var = CustomOutputVar(name="result_var")

    envs = [VariableEnvironment({}), VariableEnvironment({}), VariableEnvironment({})]

    mock_futures = [MagicMock(), MagicMock(), MagicMock()]
    mock_futures[0].result.return_value = "success"
    mock_futures[1].result.side_effect = concurrent.futures.TimeoutError()
    mock_futures[2].result.side_effect = ValueError("User logic division by zero")

    mock_pebble_pool.schedule.side_effect = mock_futures

    def fake_as_completed(futures):
        yield mock_futures[0]
        yield mock_futures[1]
        yield mock_futures[2]

    with patch("concurrent.futures.as_completed", side_effect=fake_as_completed):
        processed = processor.batch_process_sample(envs, out_var)

    assert processed[0].get("result_var") == "success"
    assert processed[1].get("result_var") == "TEST_FALLBACK"
    assert processed[2].get("result_var") == "TEST_FALLBACK"

    # Ensure internal counters tracked the failures accurately but didn't trip
    assert processor._timeout_count == 1
    assert processor._error_count == 1
    assert processor._is_broken is False


def test_start_method_defaults_to_spawn(base_config):
    """Verify the pool context defaults to 'spawn' and follows the config otherwise."""
    assert base_config.start_method == "spawn"

    with patch(
        "mmirage.core.process.processors.custom.custom_processor.multiprocessing.get_context"
    ) as mock_get_context:
        CustomProcessor(base_config)
        mock_get_context.assert_called_once_with("spawn")

        base_config.start_method = "fork"
        CustomProcessor(base_config)
        assert mock_get_context.call_args.args == ("fork",)


def test_unset_timeout_schedules_untimed(base_config, mock_pebble_pool):
    """Verify that leaving timeout_ms unset schedules rows without any timeout."""
    base_config.timeout_ms = None
    processor = CustomProcessor(base_config)
    out_var = CustomOutputVar(name="result_var")

    mock_future = MagicMock()
    mock_future.result.return_value = "success"
    mock_pebble_pool.schedule.return_value = mock_future

    with patch("concurrent.futures.as_completed", return_value=[mock_future]):
        processor.batch_process_sample([VariableEnvironment({})], out_var)

    assert mock_pebble_pool.schedule.call_args.kwargs["timeout"] is None


def test_circuit_breaker_timeout_threshold(base_config, mock_pebble_pool):
    """Verify that hitting the max_timeouts threshold actively stops the pool and fails the run."""
    processor = CustomProcessor(base_config)  # config sets max_timeouts = 2
    out_var = CustomOutputVar(name="result_var")

    envs = [VariableEnvironment({}), VariableEnvironment({})]

    mock_futures = [MagicMock(), MagicMock()]
    mock_futures[0].result.side_effect = concurrent.futures.TimeoutError()
    mock_futures[1].result.side_effect = concurrent.futures.TimeoutError()

    mock_pebble_pool.schedule.side_effect = mock_futures

    def fake_as_completed(futures):
        yield mock_futures[0]  # count = 1
        yield mock_futures[1]  # count = 2 -> TRIPS BREAKER

    with patch("concurrent.futures.as_completed", side_effect=fake_as_completed):
        with pytest.raises(RuntimeError, match="circuit breaker tripped"):
            processor.batch_process_sample(envs, out_var)

    assert processor._is_broken is True
    assert mock_pebble_pool.stop.called
    assert mock_pebble_pool.join.called


def test_circuit_breaker_fatal_crash(base_config, mock_pebble_pool):
    """Verify a dead worker trips the breaker immediately, below both thresholds."""
    processor = CustomProcessor(base_config)  # config allows 2 timeouts and 2 errors
    out_var = CustomOutputVar(name="result_var")

    mock_future = MagicMock()
    mock_future.result.side_effect = ProcessExpired("Abnormal termination", code=1)
    mock_pebble_pool.schedule.return_value = mock_future

    with patch("concurrent.futures.as_completed", return_value=[mock_future]):
        with pytest.raises(RuntimeError, match="crashed fatally"):
            processor.batch_process_sample([VariableEnvironment({})], out_var)

    assert processor._is_broken is True
    assert mock_pebble_pool.stop.called
    assert mock_pebble_pool.join.called
    # a crash is not a row-level failure: neither counter moves
    assert processor._error_count == 0
    assert processor._timeout_count == 0


def test_circuit_breaker_error_threshold(base_config, mock_pebble_pool):
    """Verify that hitting the max_errors threshold actively stops the pool and fails the run."""
    processor = CustomProcessor(base_config)  # config sets max_errors = 2
    out_var = CustomOutputVar(name="result_var")

    envs = [VariableEnvironment({}), VariableEnvironment({})]

    mock_futures = [MagicMock(), MagicMock()]
    mock_futures[0].result.side_effect = Exception("Crash 1")
    mock_futures[1].result.side_effect = Exception("Crash 2")

    mock_pebble_pool.schedule.side_effect = mock_futures

    def fake_as_completed(futures):
        yield mock_futures[0]
        yield mock_futures[1]

    with patch("concurrent.futures.as_completed", side_effect=fake_as_completed):
        with pytest.raises(RuntimeError, match="circuit breaker tripped"):
            processor.batch_process_sample(envs, out_var)

    assert processor._is_broken is True
    assert mock_pebble_pool.stop.called
    assert mock_pebble_pool.join.called


def test_worker_loads_and_executes_user_function(dummy_script, clean_worker_state):
    """Verify the worker loads the target function and returns its value."""
    worker.initialize_worker(dummy_script, "analyze_text")
    assert worker.execute_custom_function({"text": "anything"}) == "analyzed"


def test_worker_supports_local_imports(tmp_path, clean_worker_state):
    """Verify the script's own directory is importable from within the worker."""
    (tmp_path / "helper.py").write_text("VALUE = 'from_helper'")
    script = tmp_path / "importing_script.py"
    script.write_text("import helper\n\ndef run(row): return helper.VALUE")

    worker.initialize_worker(str(script), "run")
    assert worker.execute_custom_function({}) == "from_helper"


def test_worker_rejects_invalid_targets(tmp_path, dummy_script, clean_worker_state):
    """Verify each unusable target fails at initialization, not at row execution."""
    with pytest.raises(RuntimeError, match="not initialized"):
        worker.execute_custom_function({})

    with pytest.raises(AttributeError, match="not found"):
        worker.initialize_worker(dummy_script, "missing_function")

    not_callable = tmp_path / "not_callable.py"
    not_callable.write_text("analyze_text = 42")
    with pytest.raises(TypeError, match="is not callable"):
        worker.initialize_worker(str(not_callable), "analyze_text")

    with pytest.raises(FileNotFoundError):
        worker.initialize_worker(str(tmp_path / "absent.py"), "analyze_text")
