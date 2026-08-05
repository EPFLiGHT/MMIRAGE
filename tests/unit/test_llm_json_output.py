"""Tests for LLM JSON output handling: schema typing and parse-failure behavior."""

import json
import logging
from types import SimpleNamespace

import pytest

import mmirage.core.process.processors.llm.llm_processor as llm_processor_module
from mmirage.core.process.processors.llm.config import (
    LLMOutputVar,
    LLMSchemaField,
    SGLangLLMConfig,
    SGLangServerArgs,
)
from mmirage.core.process.processors.llm.llm_processor import LLMProcessor
from mmirage.core.process.variables import VariableEnvironment

FIELDS = ["relevance", "clarity", "fluency"]


def test_typed_output_schema_generates_typed_json_schema():
    output_var = LLMOutputVar(
        name="scores",
        type="llm",
        prompt="{{ text }}",
        output_type="JSON",
        output_schema={field: "int" for field in FIELDS},
    )

    model = output_var.get_output_schema()
    assert model is not None
    schema = model.model_json_schema()

    for field in FIELDS:
        assert schema["properties"][field]["type"] == "integer"
    assert set(schema["required"]) == set(FIELDS)


def test_list_output_schema_defaults_to_str_fields():
    output_var = LLMOutputVar(
        name="scores",
        type="llm",
        prompt="{{ text }}",
        output_type="JSON",
        output_schema=FIELDS,
    )

    model = output_var.get_output_schema()
    assert model is not None
    schema = model.model_json_schema()

    for field in FIELDS:
        assert schema["properties"][field]["type"] == "string"


def test_unknown_type_name_in_output_schema_raises_at_construction():
    with pytest.raises(ValueError, match="integerish"):
        LLMOutputVar(
            name="scores",
            type="llm",
            prompt="{{ text }}",
            output_type="JSON",
            output_schema={"relevance": "integerish"},
        )


def test_output_var_loads_typed_schema_from_config_dict():
    # Mirrors how config/utils.py builds output vars from YAML via dacite.
    from dacite import from_dict

    output_var = from_dict(
        LLMOutputVar,
        {
            "name": "scores",
            "type": "llm",
            "prompt": "{{ text }}",
            "output_type": "JSON",
            "output_schema": {field: "int" for field in FIELDS},
        },
    )

    model = output_var.get_output_schema()
    assert model is not None
    assert model.model_json_schema()["properties"]["clarity"]["type"] == "integer"


def _make_processor(
    monkeypatch, generated_text: str, sampling_params_seen: list | None = None
) -> LLMProcessor:
    """Build an LLMProcessor whose engine returns `generated_text` for every prompt."""

    class FakeEngine:
        def __init__(self, **_kwargs):
            return None

        def generate(self, prompt, sampling_params=None, **_kwargs):
            if sampling_params_seen is not None:
                sampling_params_seen.append(sampling_params)
            return [
                {"text": generated_text, "meta_info": {}}
                for _ in (prompt if isinstance(prompt, list) else [prompt])
            ]

        def shutdown(self):
            return None

    class FakeTokenizer:
        def apply_chat_template(
            self, user_prompt, tokenize=False, add_generation_prompt=True
        ):
            return user_prompt[0]["content"]

    monkeypatch.setattr(
        llm_processor_module, "sgl", SimpleNamespace(Engine=FakeEngine), raising=False
    )
    monkeypatch.setattr(llm_processor_module, "SGLANG_AVAILABLE", True)
    monkeypatch.setattr(
        llm_processor_module.AutoTokenizer,
        "from_pretrained",
        classmethod(lambda *args, **kwargs: FakeTokenizer()),
    )

    config = SGLangLLMConfig(
        type="llm",
        server_args=SGLangServerArgs(model_path="dummy-model"),
    )
    return LLMProcessor(config)


def test_json_parse_failure_logs_raw_output_and_falls_back_to_empty_dict(
    monkeypatch, caplog
):
    truncated = '{"relevance": "1", "clarity": "the answer mentions sev'
    processor = _make_processor(monkeypatch, truncated)

    output_var = LLMOutputVar(
        name="scores",
        type="llm",
        prompt="{{ text }}",
        output_type="JSON",
        output_schema=FIELDS,
    )
    batch = [VariableEnvironment({"text": "some input"})]

    with caplog.at_level(logging.WARNING, logger=llm_processor_module.__name__):
        results = processor.batch_process_sample(batch, output_var)

    assert results[0].get("scores") == {}
    failure_logs = [
        record.getMessage()
        for record in caplog.records
        if "scores" in record.getMessage() and truncated in record.getMessage()
    ]
    assert failure_logs, "expected a warning containing the raw model output"


def test_valid_json_output_is_parsed(monkeypatch):
    valid = json.dumps({field: "1" for field in FIELDS})
    processor = _make_processor(monkeypatch, valid)

    output_var = LLMOutputVar(
        name="scores",
        type="llm",
        prompt="{{ text }}",
        output_type="JSON",
        output_schema=FIELDS,
    )
    batch = [VariableEnvironment({"text": "some input"})]

    results = processor.batch_process_sample(batch, output_var)

    assert results[0].get("scores") == {field: "1" for field in FIELDS}


CONSTRAINED_SCHEMA = {field: {"type": "int", "min": 0, "max": 3} for field in FIELDS}


def _json_output_var(output_schema) -> LLMOutputVar:
    return LLMOutputVar(
        name="scores",
        type="llm",
        prompt="{{ text }}",
        output_type="JSON",
        output_schema=output_schema,
    )


def test_constrained_field_generates_min_max_in_json_schema():
    model = _json_output_var(CONSTRAINED_SCHEMA).get_output_schema()
    assert model is not None
    schema = model.model_json_schema()

    for field in FIELDS:
        assert schema["properties"][field]["type"] == "integer"
        assert schema["properties"][field]["minimum"] == 0
        assert schema["properties"][field]["maximum"] == 3
    assert set(schema["required"]) == set(FIELDS)


def test_constraint_with_single_bound():
    model = _json_output_var(
        {"relevance": {"type": "int", "min": 0}}
    ).get_output_schema()
    assert model is not None
    prop = model.model_json_schema()["properties"]["relevance"]
    assert prop["minimum"] == 0
    assert "maximum" not in prop


@pytest.mark.parametrize(
    ("schema", "match"),
    [
        ({"relevance": {"type": "int", "min": 0, "clamp": True}}, "Unknown key"),
        ({"relevance": {"min": 0, "max": 3}}, "missing required key 'type'"),
        ({"relevance": {"type": "integerish"}}, "Unsupported type 'integerish'"),
        ({"relevance": {"type": "str", "min": 0}}, "only allowed for numeric"),
        ({"relevance": {"type": "int", "min": 3, "max": 0}}, "cannot be greater than"),
        ({"relevance": {"type": "int", "min": "zero"}}, "must be a number"),
        ({"relevance": {"type": "int", "min": 0, "max": [3]}}, "must be a number"),
        ({"relevance": {"type": "int", "min": True, "max": 3}}, "must be a number"),
        ({"relevance": {"type": "int", "min": 0.5}}, "must be a whole number"),
        ({"relevance": {"type": "int", "min": "0.5"}}, "must be a whole number"),
    ],
    ids=[
        "unknown-key",
        "missing-type",
        "bad-type",
        "bound-on-str",
        "min-gt-max",
        "non-numeric-string-bound",
        "list-bound",
        "bool-bound",
        "fractional-bound-on-int",
        "fractional-string-bound-on-int",
    ],
)
def test_invalid_nested_schema_raises(schema, match):
    with pytest.raises(ValueError, match=match):
        _json_output_var(schema)


def test_numeric_string_bounds_are_accepted():
    model = _json_output_var(
        {"relevance": {"type": "int", "min": "0", "max": "3"}}
    ).get_output_schema()
    assert model is not None
    prop = model.model_json_schema()["properties"]["relevance"]
    assert prop["minimum"] == 0
    assert prop["maximum"] == 3


def test_string_bounds_are_coerced_to_the_field_numeric_type():
    int_field = LLMSchemaField(type="int", min="0", max="3")
    assert int_field.min == 0 and isinstance(int_field.min, int)
    assert int_field.max == 3 and isinstance(int_field.max, int)

    float_field = LLMSchemaField(type="float", min="-1.5")
    assert float_field.min == -1.5


def test_env_var_style_string_bounds_load_from_config_dict():
    from dacite import from_dict

    output_var = from_dict(
        LLMOutputVar,
        {
            "name": "scores",
            "type": "llm",
            "prompt": "{{ text }}",
            "output_type": "JSON",
            "output_schema": {"relevance": {"type": "int", "min": "0", "max": "3"}},
        },
    )

    model = output_var.get_output_schema()
    assert model is not None
    assert model.model_json_schema()["properties"]["relevance"]["maximum"] == 3


def test_output_var_loads_constrained_schema_from_config_dict():
    from dacite import from_dict

    output_var = from_dict(
        LLMOutputVar,
        {
            "name": "scores",
            "type": "llm",
            "prompt": "{{ text }}",
            "output_type": "JSON",
            "output_schema": CONSTRAINED_SCHEMA,
        },
    )

    model = output_var.get_output_schema()
    assert model is not None
    assert model.model_json_schema()["properties"]["clarity"]["maximum"] == 3


def test_bounds_reach_the_engine_as_constrained_decoding_schema(monkeypatch):
    sampling_params_seen: list = []
    _make_processor(
        monkeypatch,
        json.dumps({field: 1 for field in FIELDS}),
        sampling_params_seen,
    ).batch_process_sample(
        [VariableEnvironment({"text": "some input"})],
        _json_output_var(CONSTRAINED_SCHEMA),
    )

    assert sampling_params_seen, "engine was never called"
    schema = json.loads(sampling_params_seen[0]["json_schema"])
    assert schema["properties"]["clarity"] == {
        "type": "integer",
        "minimum": 0,
        "maximum": 3,
        "title": "Clarity",
    }


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        (CONSTRAINED_SCHEMA, True),
        ({"relevance": {"type": "int"}}, False),
        ({"relevance": {"type": "int", "min": None}}, False),
        ({"relevance": "int"}, False),
        (FIELDS, False),
    ],
    ids=["bounded", "no-keys", "null-bound", "shorthand", "list-form"],
)
def test_has_schema_constraints_agrees_with_the_built_model(schema, expected):
    output_var = _json_output_var(schema)
    model = output_var.get_output_schema()
    assert model is not None
    model_has_bounds = any(
        "minimum" in prop or "maximum" in prop
        for prop in model.model_json_schema()["properties"].values()
    )

    assert output_var.has_schema_constraints() is expected
    assert model_has_bounds is expected


def test_missing_fields_are_reported_without_the_parent_dict(monkeypatch, caplog):
    processor = _make_processor(monkeypatch, json.dumps({"relevance": 1}))
    batch = [VariableEnvironment({"text": "some input"})]

    with caplog.at_level(logging.WARNING, logger=llm_processor_module.__name__):
        results = processor.batch_process_sample(
            batch, _json_output_var(CONSTRAINED_SCHEMA)
        )

    assert results[0].get("scores") == {"relevance": 1}
    reported_errors = caplog.records[0].getMessage().split("keeping parsed value.")[1]
    assert "clarity: Field required" in reported_errors
    assert "relevance" not in reported_errors


def test_out_of_range_value_logs_warning_and_keeps_value(monkeypatch, caplog):
    scores = {field: 1 for field in FIELDS} | {"fluency": -1}
    processor = _make_processor(monkeypatch, json.dumps(scores))

    output_var = _json_output_var(CONSTRAINED_SCHEMA)
    batch = [VariableEnvironment({"text": "some input"})]

    with caplog.at_level(logging.WARNING, logger=llm_processor_module.__name__):
        results = processor.batch_process_sample(batch, output_var)

    assert results[0].get("scores") == scores
    violation_logs = [
        record.getMessage()
        for record in caplog.records
        if "scores" in record.getMessage() and "fluency" in record.getMessage()
    ]
    assert violation_logs, "expected a constraint-violation warning naming the field"


def test_unconstrained_schema_does_not_validate_parsed_output(monkeypatch, caplog):
    processor = _make_processor(monkeypatch, json.dumps({"relevance": 1}))

    output_var = _json_output_var(["relevance"])
    batch = [VariableEnvironment({"text": "some input"})]

    with caplog.at_level(logging.WARNING, logger=llm_processor_module.__name__):
        results = processor.batch_process_sample(batch, output_var)

    assert results[0].get("scores") == {"relevance": 1}
    assert not caplog.records


def test_in_range_values_do_not_warn(monkeypatch, caplog):
    scores = {field: 1 for field in FIELDS}
    processor = _make_processor(monkeypatch, json.dumps(scores))

    output_var = _json_output_var(CONSTRAINED_SCHEMA)
    batch = [VariableEnvironment({"text": "some input"})]

    with caplog.at_level(logging.WARNING, logger=llm_processor_module.__name__):
        results = processor.batch_process_sample(batch, output_var)

    assert results[0].get("scores") == scores
    assert not caplog.records
