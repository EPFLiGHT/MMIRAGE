from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pytest
from pydantic import ValidationError

from mmirage.core.process.structured_outputs import (
    DspyBackend,
    ExecutionAST,
    StructuredOutputBackend,
    StructuredTaskRunner,
    TaskSpecAST,
    build_pydantic_model,
    compile_task,
    normalize_backend_output,
    parse_task_spec_yaml,
)


SAMPLE_SPEC_YAML = """
input_schema:
  question:
    type: string
    description: User question
  patient_age:
    type: integer
output_schema:
  answer:
    type: string
  confidence:
    type: number
  tags:
    type: array
    items: string
  metadata:
    type: object
    properties:
      severity:
        type: string
      urgent:
        type: boolean

dataset_mapping:
  inputs:
    question: text
    patient_age: age
  outputs:
    result.answer: answer
    result.confidence: confidence
    result.flags.urgent: metadata.urgent

execution:
  backend: dspy
  instruction: Generate a concise, structured answer.
  temperature: 0.0
  max_tokens: 128
"""


class FakePrediction:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeProgram:
    def __init__(self, signature):
        self.signature = signature
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return FakePrediction(
            answer="Stable patient.",
            confidence=0.83,
            tags=["triage", "stable"],
            metadata={"severity": "low", "urgent": False},
        )


class FakeDSPY:
    class Signature:
        pass

    @staticmethod
    def InputField(desc: str = ""):
        return {"kind": "input", "desc": desc}

    @staticmethod
    def OutputField(desc: str = ""):
        return {"kind": "output", "desc": desc}

    @staticmethod
    def Predict(signature):
        return FakeProgram(signature)


@dataclass
class EchoProgram:
    pass


class EchoBackend(StructuredOutputBackend):
    def compile(self, ast: TaskSpecAST) -> EchoProgram:
        return EchoProgram()

    def infer(self, backend_program: EchoProgram, inputs: Dict[str, Any]) -> Any:
        assert isinstance(backend_program, EchoProgram)
        assert "question" in inputs
        return {
            "answer": f"Answer to: {inputs['question']}",
            "confidence": 0.91,
            "tags": ["ok"],
            "metadata": {"severity": "medium", "urgent": False},
        }


def test_parse_yaml_to_ast_and_validate_supported_types(tmp_path: Path):
    spec_file = tmp_path / "task.yaml"
    spec_file.write_text(SAMPLE_SPEC_YAML)

    ast = parse_task_spec_yaml(spec_file)

    assert ast.execution == ExecutionAST(
        backend="dspy",
        instruction="Generate a concise, structured answer.",
        model=None,
        temperature=0.0,
        max_tokens=128,
    )
    assert ast.input_schema.fields["question"].type_spec.kind == "string"
    assert ast.input_schema.fields["patient_age"].type_spec.kind == "integer"

    output_meta = ast.output_schema.fields["metadata"].type_spec
    assert output_meta.kind == "object"
    assert output_meta.properties is not None
    assert output_meta.properties["urgent"].kind == "boolean"


def test_parser_rejects_nested_object_deeper_than_one_level():
    bad_yaml = """
input_schema:
  x: string
output_schema:
  nested:
    type: object
    properties:
      level1:
        type: object
        properties:
          level2:
            type: string
dataset_mapping:
  inputs:
    x: x
  outputs: {}
execution:
  backend: dspy
"""
    with pytest.raises(ValueError, match="Nested object"):
        parse_task_spec_yaml(bad_yaml)


def test_build_model_and_normalize_output():
    ast = parse_task_spec_yaml(SAMPLE_SPEC_YAML)
    model = build_pydantic_model(ast.output_schema, model_name="TestOut")

    normalized = normalize_backend_output(
        '{"answer":"ok","confidence":0.5,"tags":["a"],"metadata":{"severity":"low","urgent":true}}',
        ast.output_schema,
    )
    validated = model.model_validate(normalized)

    assert validated.answer == "ok"
    assert validated.metadata.urgent is True


def test_pydantic_validation_fails_on_wrong_type():
    ast = parse_task_spec_yaml(SAMPLE_SPEC_YAML)
    model = build_pydantic_model(ast.output_schema, model_name="BadOut")

    with pytest.raises(ValidationError):
        model.model_validate(
            {
                "answer": "ok",
                "confidence": "not-a-number",
                "tags": ["x"],
                "metadata": {"severity": "low", "urgent": False},
            }
        )


def test_dspy_backend_compiles_signature_and_runs_infer():
    ast = parse_task_spec_yaml(SAMPLE_SPEC_YAML)
    backend = DspyBackend(dspy_module=FakeDSPY)

    compiled = compile_task(ast, backend)

    assert compiled.backend_program.signature.__name__ == "MMIRAGEStructuredSignature"
    assert "question" in compiled.backend_program.signature.__annotations__
    assert "answer" in compiled.backend_program.signature.__annotations__

    raw = backend.infer(compiled.backend_program, {"question": "q", "patient_age": 33})
    assert raw.answer == "Stable patient."


def test_end_to_end_runner_maps_fields_back_into_dataset():
    ast = parse_task_spec_yaml(SAMPLE_SPEC_YAML)
    backend = EchoBackend()
    compiled = compile_task(ast, backend)

    runner = StructuredTaskRunner(compiled, backend)
    sample = {"text": "What is the status?", "age": 44, "id": "abc"}

    out = runner.run_sample(sample)

    assert out["id"] == "abc"
    assert out["result"]["answer"] == "Answer to: What is the status?"
    assert out["result"]["confidence"] == 0.91
    assert out["result"]["flags"]["urgent"] is False
