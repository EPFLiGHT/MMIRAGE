"""Tests for LLM JSON output handling: schema typing and parse-failure behavior."""

import json
import logging
from types import SimpleNamespace

import pytest

import mmirage.core.process.processors.llm.llm_processor as llm_processor_module
from mmirage.core.process.processors.llm.config import (
    LLMOutputVar,
    SGLangLLMConfig,
    SGLangServerArgs,
)
from mmirage.core.process.processors.llm.llm_processor import LLMProcessor
from mmirage.core.process.variables import VariableEnvironment

PHQ8_FIELDS = [
    "NoInterest",
    "Depressed",
    "Sleep",
    "Tired",
    "Appetite",
    "Failure",
    "Concentrating",
    "Moving",
]


def test_typed_output_schema_generates_typed_json_schema():
    output_var = LLMOutputVar(
        name="phq8_scores",
        type="llm",
        prompt="{{ transcript }}",
        output_type="JSON",
        output_schema={field: "int" for field in PHQ8_FIELDS},
    )

    model = output_var.get_output_schema()
    assert model is not None
    schema = model.model_json_schema()

    for field in PHQ8_FIELDS:
        assert schema["properties"][field]["type"] == "integer"
    assert set(schema["required"]) == set(PHQ8_FIELDS)


def test_list_output_schema_defaults_to_str_fields():
    output_var = LLMOutputVar(
        name="phq8_scores",
        type="llm",
        prompt="{{ transcript }}",
        output_type="JSON",
        output_schema=PHQ8_FIELDS,
    )

    model = output_var.get_output_schema()
    assert model is not None
    schema = model.model_json_schema()

    for field in PHQ8_FIELDS:
        assert schema["properties"][field]["type"] == "string"


def test_unknown_type_name_in_output_schema_raises():
    output_var = LLMOutputVar(
        name="phq8_scores",
        type="llm",
        prompt="{{ transcript }}",
        output_type="JSON",
        output_schema={"NoInterest": "integerish"},
    )

    with pytest.raises(ValueError, match="integerish"):
        output_var.get_output_schema()


def test_output_var_loads_typed_schema_from_config_dict():
    # Mirrors how config/utils.py builds output vars from YAML via dacite.
    from dacite import from_dict

    output_var = from_dict(
        LLMOutputVar,
        {
            "name": "phq8_scores",
            "type": "llm",
            "prompt": "{{ transcript }}",
            "output_type": "JSON",
            "output_schema": {field: "int" for field in PHQ8_FIELDS},
        },
    )

    model = output_var.get_output_schema()
    assert model is not None
    assert model.model_json_schema()["properties"]["Sleep"]["type"] == "integer"


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
    # Reproduces the PHQ-8 bug: str-typed constrained decoding rambles inside a
    # string value, generation is truncated at max_new_tokens, and the invalid
    # JSON silently becomes {}.
    truncated = '{"NoInterest": "0", "Depressed": "the participant mentions feel'
    processor = _make_processor(monkeypatch, truncated)

    output_var = LLMOutputVar(
        name="phq8_scores",
        type="llm",
        prompt="{{ transcript }}",
        output_type="JSON",
        output_schema=PHQ8_FIELDS,
    )
    batch = [VariableEnvironment({"transcript": "hi ellie"})]

    with caplog.at_level(logging.WARNING, logger=llm_processor_module.__name__):
        results = processor.batch_process_sample(batch, output_var)

    assert results[0].get("phq8_scores") == {}
    failure_logs = [
        record.getMessage()
        for record in caplog.records
        if "phq8_scores" in record.getMessage() and truncated in record.getMessage()
    ]
    assert failure_logs, "expected a warning containing the raw model output"


def test_valid_json_output_is_parsed(monkeypatch):
    valid = json.dumps({field: "1" for field in PHQ8_FIELDS})
    processor = _make_processor(monkeypatch, valid)

    output_var = LLMOutputVar(
        name="phq8_scores",
        type="llm",
        prompt="{{ transcript }}",
        output_type="JSON",
        output_schema=PHQ8_FIELDS,
    )
    batch = [VariableEnvironment({"transcript": "hi ellie"})]

    results = processor.batch_process_sample(batch, output_var)

    assert results[0].get("phq8_scores") == {field: "1" for field in PHQ8_FIELDS}


PHQ8_CONSTRAINED_SCHEMA = {
    field: {"type": "int", "min": 0, "max": 3} for field in PHQ8_FIELDS
}


def _phq8_output_var(output_schema) -> LLMOutputVar:
    return LLMOutputVar(
        name="phq8_scores",
        type="llm",
        prompt="{{ transcript }}",
        output_type="JSON",
        output_schema=output_schema,
    )


def test_constrained_field_generates_min_max_in_json_schema():
    model = _phq8_output_var(PHQ8_CONSTRAINED_SCHEMA).get_output_schema()
    assert model is not None
    schema = model.model_json_schema()

    for field in PHQ8_FIELDS:
        assert schema["properties"][field]["type"] == "integer"
        assert schema["properties"][field]["minimum"] == 0
        assert schema["properties"][field]["maximum"] == 3
    assert set(schema["required"]) == set(PHQ8_FIELDS)


def test_constraint_with_single_bound():
    model = _phq8_output_var(
        {"NoInterest": {"type": "int", "min": 0}}
    ).get_output_schema()
    assert model is not None
    prop = model.model_json_schema()["properties"]["NoInterest"]
    assert prop["minimum"] == 0
    assert "maximum" not in prop


@pytest.mark.parametrize(
    ("schema", "match"),
    [
        ({"NoInterest": {"type": "int", "min": 0, "clamp": True}}, "Unknown key"),
        ({"NoInterest": {"min": 0, "max": 3}}, "missing required key 'type'"),
        ({"NoInterest": {"type": "integerish"}}, "Unsupported type 'integerish'"),
        ({"NoInterest": {"type": "str", "min": 0}}, "only allowed for numeric"),
        ({"NoInterest": {"type": "int", "min": 3, "max": 0}}, "cannot be greater than"),
    ],
    ids=["unknown-key", "missing-type", "bad-type", "bound-on-str", "min-gt-max"],
)
def test_invalid_nested_schema_raises(schema, match):
    with pytest.raises(ValueError, match=match):
        _phq8_output_var(schema).get_output_schema()


def test_output_var_loads_constrained_schema_from_config_dict():
    from dacite import from_dict

    output_var = from_dict(
        LLMOutputVar,
        {
            "name": "phq8_scores",
            "type": "llm",
            "prompt": "{{ transcript }}",
            "output_type": "JSON",
            "output_schema": PHQ8_CONSTRAINED_SCHEMA,
        },
    )

    model = output_var.get_output_schema()
    assert model is not None
    assert model.model_json_schema()["properties"]["Sleep"]["maximum"] == 3


def test_bounds_reach_the_engine_as_constrained_decoding_schema(monkeypatch):
    sampling_params_seen: list = []
    _make_processor(
        monkeypatch,
        json.dumps({field: 1 for field in PHQ8_FIELDS}),
        sampling_params_seen,
    ).batch_process_sample(
        [VariableEnvironment({"transcript": "hi ellie"})],
        _phq8_output_var(PHQ8_CONSTRAINED_SCHEMA),
    )

    assert sampling_params_seen, "engine was never called"
    schema = json.loads(sampling_params_seen[0]["json_schema"])
    assert schema["properties"]["Sleep"] == {
        "type": "integer",
        "minimum": 0,
        "maximum": 3,
        "title": "Sleep",
    }


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        (PHQ8_CONSTRAINED_SCHEMA, True),
        ({"NoInterest": {"type": "int"}}, False),
        ({"NoInterest": {"type": "int", "min": None}}, False),
        ({"NoInterest": "int"}, False),
        (PHQ8_FIELDS, False),
    ],
    ids=["bounded", "no-keys", "null-bound", "shorthand", "list-form"],
)
def test_has_schema_constraints_agrees_with_the_built_model(schema, expected):
    output_var = _phq8_output_var(schema)
    model = output_var.get_output_schema()
    assert model is not None
    model_has_bounds = any(
        "minimum" in prop or "maximum" in prop
        for prop in model.model_json_schema()["properties"].values()
    )

    assert output_var.has_schema_constraints() is expected
    assert model_has_bounds is expected


def test_missing_fields_are_reported_without_the_parent_dict(monkeypatch, caplog):
    processor = _make_processor(monkeypatch, json.dumps({"NoInterest": 1}))
    batch = [VariableEnvironment({"transcript": "hi ellie"})]

    with caplog.at_level(logging.WARNING, logger=llm_processor_module.__name__):
        results = processor.batch_process_sample(
            batch, _phq8_output_var(PHQ8_CONSTRAINED_SCHEMA)
        )

    assert results[0].get("phq8_scores") == {"NoInterest": 1}
    reported_errors = caplog.records[0].getMessage().split("keeping parsed value.")[1]
    assert "Sleep: Field required" in reported_errors
    assert "NoInterest" not in reported_errors


def test_out_of_range_value_logs_warning_and_keeps_value(monkeypatch, caplog):
    scores = {field: 1 for field in PHQ8_FIELDS} | {"Appetite": -1}
    processor = _make_processor(monkeypatch, json.dumps(scores))

    output_var = _phq8_output_var(PHQ8_CONSTRAINED_SCHEMA)
    batch = [VariableEnvironment({"transcript": "hi ellie"})]

    with caplog.at_level(logging.WARNING, logger=llm_processor_module.__name__):
        results = processor.batch_process_sample(batch, output_var)

    assert results[0].get("phq8_scores") == scores
    violation_logs = [
        record.getMessage()
        for record in caplog.records
        if "phq8_scores" in record.getMessage() and "Appetite" in record.getMessage()
    ]
    assert violation_logs, "expected a constraint-violation warning naming the field"


def test_unconstrained_schema_does_not_validate_parsed_output(monkeypatch, caplog):
    processor = _make_processor(monkeypatch, json.dumps({"NoInterest": 1}))

    output_var = _phq8_output_var(["NoInterest"])
    batch = [VariableEnvironment({"transcript": "hi ellie"})]

    with caplog.at_level(logging.WARNING, logger=llm_processor_module.__name__):
        results = processor.batch_process_sample(batch, output_var)

    assert results[0].get("phq8_scores") == {"NoInterest": 1}
    assert not caplog.records


def test_in_range_values_do_not_warn(monkeypatch, caplog):
    scores = {field: 1 for field in PHQ8_FIELDS}
    processor = _make_processor(monkeypatch, json.dumps(scores))

    output_var = _phq8_output_var(PHQ8_CONSTRAINED_SCHEMA)
    batch = [VariableEnvironment({"transcript": "hi ellie"})]

    with caplog.at_level(logging.WARNING, logger=llm_processor_module.__name__):
        results = processor.batch_process_sample(batch, output_var)

    assert results[0].get("phq8_scores") == scores
    assert not caplog.records
