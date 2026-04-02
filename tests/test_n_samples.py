"""Unit tests for n_samples batch expansion in rewrite_batch."""

from typing import Any, Dict, List
from unittest.mock import MagicMock

from mmirage.core.process.variables import VariableEnvironment
from mmirage.shard_process import rewrite_batch


def _make_mapper(output_name: str = "out") -> MagicMock:
    """Return a mock MMIRAGEMapper that appends '_processed' to the 'text' input."""

    def fake_rewrite_batch(batch, image_base_path=None):
        size = len(next(iter(batch.values())))
        return [
            VariableEnvironment({output_name: batch["text"][i] + "_processed"})
            for i in range(size)
        ]

    mapper = MagicMock()
    mapper.validate_vars.return_value = True
    mapper.rewrite_batch.side_effect = fake_rewrite_batch
    return mapper


def _make_renderer(output_name: str = "out") -> MagicMock:
    """Return a mock TemplateRenderer that returns {output_name: [values...]}."""

    def fake_batch_render(envs: List[VariableEnvironment]) -> Dict[str, List[Any]]:
        return {output_name: [e.get(output_name) for e in envs]}

    renderer = MagicMock()
    renderer.batch_render.side_effect = fake_batch_render
    return renderer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_n_samples_1_is_identity():
    """n_samples=1 should behave exactly like the original pipeline."""
    batch = {"text": ["a", "b", "c"]}
    result = rewrite_batch(batch, _make_mapper(), _make_renderer(), n_samples=1)
    assert result["out"] == ["a_processed", "b_processed", "c_processed"]


def test_n_samples_expands_rows():
    """n_samples=3 should produce 3 rows per input row."""
    batch = {"text": ["a", "b"]}
    result = rewrite_batch(batch, _make_mapper(), _make_renderer(), n_samples=3)
    assert len(result["out"]) == 6


def test_n_samples_row_order():
    """Each input row should appear N times consecutively, not interleaved."""
    batch = {"text": ["x", "y"]}
    result = rewrite_batch(batch, _make_mapper(), _make_renderer(), n_samples=3)
    # Expected order: x x x y y y (not x y x y x y)
    assert result["out"] == [
        "x_processed", "x_processed", "x_processed",
        "y_processed", "y_processed", "y_processed",
    ]


def test_preserve_columns_attaches_original():
    """preserve_columns=True should re-attach original columns scaled up."""
    batch = {"text": ["a", "b"], "label": [0, 1]}
    result = rewrite_batch(
        batch, _make_mapper(), _make_renderer(), n_samples=2, preserve_columns=True
    )
    assert result["label"] == [0, 0, 1, 1]
    assert len(result["out"]) == 4


def test_preserve_columns_false_excludes_original():
    """preserve_columns=False should not include original columns in output."""
    batch = {"text": ["a", "b"], "label": [0, 1]}
    result = rewrite_batch(
        batch, _make_mapper(), _make_renderer(), n_samples=2, preserve_columns=False
    )
    assert "label" not in result


def test_preserve_columns_does_not_overwrite_rendered():
    """preserve_columns should not overwrite a column already in rendered output."""
    # If the output schema produces a column with the same name as an input column,
    # the rendered value must win.
    batch = {"text": ["a"], "out": ["original"]}
    result = rewrite_batch(
        batch, _make_mapper(), _make_renderer(), n_samples=2, preserve_columns=True
    )
    assert result["out"] == ["a_processed", "a_processed"]
