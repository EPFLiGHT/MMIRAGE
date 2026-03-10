"""Structured output MVP for MMIRAGE with optional DSPy backend.

This module keeps a MMIRAGE-specific AST as source of truth and provides:
- YAML task spec parsing into AST dataclasses
- Backend compilation and inference abstraction
- Optional DSPy backend implementation
- Output normalization + Pydantic validation
- Dataset input/output field mapping helpers
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import abc
import json
from typing import Any, Dict, List, Literal, Mapping, MutableMapping, Optional, Sequence, Tuple, Type, Union

import jmespath
from pydantic import BaseModel, ValidationError, create_model
import yaml

ScalarType = Literal["string", "integer", "number", "boolean"]
SchemaType = Literal["string", "integer", "number", "boolean", "array", "object"]


@dataclass(frozen=True)
class TypeSpecAST:
    """Type information for a schema field."""

    kind: SchemaType
    items_kind: Optional[ScalarType] = None
    properties: Optional[Dict[str, "TypeSpecAST"]] = None


@dataclass(frozen=True)
class FieldSpecAST:
    """Field definition in an input/output schema."""

    name: str
    type_spec: TypeSpecAST
    description: str = ""


@dataclass(frozen=True)
class SchemaAST:
    """Schema object for inputs or outputs."""

    fields: Dict[str, FieldSpecAST]


@dataclass(frozen=True)
class DatasetMappingAST:
    """Mapping between dataset fields and structured task fields."""

    inputs: Dict[str, str]
    outputs: Dict[str, str]


@dataclass(frozen=True)
class ExecutionAST:
    """Execution metadata used by backends."""

    backend: str = "dspy"
    instruction: str = ""
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


@dataclass(frozen=True)
class TaskSpecAST:
    """Top-level structured task specification."""

    input_schema: SchemaAST
    output_schema: SchemaAST
    dataset_mapping: DatasetMappingAST
    execution: ExecutionAST


@dataclass(frozen=True)
class CompiledStructuredTask:
    """Compiled task holding AST + backend-specific program + output validator."""

    ast: TaskSpecAST
    backend_program: Any
    output_model: Type[BaseModel]


class StructuredOutputBackend(abc.ABC):
    """Backend interface for structured output inference."""

    @abc.abstractmethod
    def compile(self, ast: TaskSpecAST) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def infer(self, backend_program: Any, inputs: Dict[str, Any]) -> Any:
        raise NotImplementedError


@dataclass(frozen=True)
class DspyCompiledProgram:
    """DSPy-specific compiled artifact."""

    signature: Any
    program: Any
    call_config: Dict[str, Any]


class DspyBackend(StructuredOutputBackend):
    """DSPy backend implementation.

    DSPy is imported lazily so this module remains usable without DSPy installed.
    """

    def __init__(self, dspy_module: Any = None):
        self._dspy = dspy_module

    def _load_dspy(self) -> Any:
        if self._dspy is not None:
            return self._dspy
        try:
            import dspy  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "DSPy is required for DspyBackend. Install `dspy` to use this backend."
            ) from exc
        self._dspy = dspy
        return dspy

    def compile(self, ast: TaskSpecAST) -> DspyCompiledProgram:
        dspy = self._load_dspy()

        namespace: Dict[str, Any] = {"__doc__": ast.execution.instruction or "MMIRAGE structured task"}
        annotations: Dict[str, Any] = {}

        for name, field in ast.input_schema.fields.items():
            annotations[name] = _to_python_type(field.type_spec)
            namespace[name] = dspy.InputField(desc=field.description)

        for name, field in ast.output_schema.fields.items():
            annotations[name] = _to_python_type(field.type_spec)
            namespace[name] = dspy.OutputField(desc=field.description)

        namespace["__annotations__"] = annotations
        signature = type("MMIRAGEStructuredSignature", (dspy.Signature,), namespace)
        program = dspy.Predict(signature)

        call_config: Dict[str, Any] = {}
        if ast.execution.temperature is not None:
            call_config["temperature"] = ast.execution.temperature
        if ast.execution.max_tokens is not None:
            call_config["max_tokens"] = ast.execution.max_tokens

        return DspyCompiledProgram(signature=signature, program=program, call_config=call_config)

    def infer(self, backend_program: DspyCompiledProgram, inputs: Dict[str, Any]) -> Any:
        if backend_program.call_config:
            try:
                return backend_program.program(**inputs, config=backend_program.call_config)
            except TypeError:
                # Fallback if DSPy version does not accept config kwarg.
                return backend_program.program(**inputs)
        return backend_program.program(**inputs)


def parse_task_spec(spec_data: Mapping[str, Any]) -> TaskSpecAST:
    """Parse validated task specification dictionary into AST."""

    required_keys = ["input_schema", "output_schema", "dataset_mapping", "execution"]
    for key in required_keys:
        if key not in spec_data:
            raise ValueError(f"Missing required top-level key: '{key}'")

    input_schema = _parse_schema(spec_data["input_schema"], section="input_schema")
    output_schema = _parse_schema(spec_data["output_schema"], section="output_schema")
    dataset_mapping = _parse_dataset_mapping(
        spec_data["dataset_mapping"], input_schema=input_schema, output_schema=output_schema
    )
    execution = _parse_execution(spec_data["execution"])

    return TaskSpecAST(
        input_schema=input_schema,
        output_schema=output_schema,
        dataset_mapping=dataset_mapping,
        execution=execution,
    )


def parse_task_spec_yaml(yaml_input: Union[str, Path]) -> TaskSpecAST:
    """Parse a YAML string/path task spec into AST dataclasses."""

    raw: Any
    path: Optional[Path] = None
    if isinstance(yaml_input, (str, Path)):
        try:
            candidate = Path(yaml_input)
            if candidate.exists():
                path = candidate
        except OSError:
            path = None
    if path is not None:
        raw = yaml.safe_load(path.read_text()) or {}
    else:
        raw = yaml.safe_load(str(yaml_input)) or {}

    if not isinstance(raw, dict):
        raise ValueError("Task spec YAML must decode to an object at top level.")

    return parse_task_spec(raw)


def compile_task(ast: TaskSpecAST, backend: StructuredOutputBackend) -> CompiledStructuredTask:
    """Compile AST to backend program and create output validator model."""

    backend_program = backend.compile(ast)
    output_model = build_pydantic_model(ast.output_schema, model_name="MMIRAGEStructuredOutput")
    return CompiledStructuredTask(ast=ast, backend_program=backend_program, output_model=output_model)


def extract_inputs(sample: Mapping[str, Any], mapping: DatasetMappingAST) -> Dict[str, Any]:
    """Extract backend inputs from dataset sample using JMESPath expressions."""

    inputs: Dict[str, Any] = {}
    for input_name, jmes_expr in mapping.inputs.items():
        value = jmespath.search(jmes_expr, sample)
        if value is None:
            raise ValueError(
                f"Input mapping for '{input_name}' did not resolve any value. "
                f"Expression: '{jmes_expr}'"
            )
        inputs[input_name] = value
    return inputs


def normalize_backend_output(raw_output: Any, output_schema: SchemaAST) -> Dict[str, Any]:
    """Normalize backend output into a dictionary compatible with output schema."""

    output_fields = tuple(output_schema.fields.keys())

    if isinstance(raw_output, BaseModel):
        raw_dict = raw_output.model_dump()
    elif isinstance(raw_output, dict):
        raw_dict = dict(raw_output)
    elif isinstance(raw_output, str):
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            raise ValueError("Backend output is a string but not valid JSON.") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Backend JSON output must be an object.")
        raw_dict = parsed
    else:
        raw_dict = {
            name: getattr(raw_output, name)
            for name in output_fields
            if hasattr(raw_output, name)
        }

    normalized: Dict[str, Any] = {}
    for field_name, field in output_schema.fields.items():
        if field_name not in raw_dict:
            continue
        normalized[field_name] = _normalize_value(raw_dict[field_name], field.type_spec)

    return normalized


def validate_output(output_model: Type[BaseModel], output_data: Dict[str, Any]) -> BaseModel:
    """Validate normalized output data with Pydantic."""

    return output_model.model_validate(output_data)


def map_outputs_to_dataset(
    sample: Mapping[str, Any],
    validated_output: BaseModel,
    mapping: DatasetMappingAST,
) -> Dict[str, Any]:
    """Map validated output fields to dataset keys and return updated sample."""

    out = dict(sample)
    output_dict = validated_output.model_dump()

    if mapping.outputs:
        for dataset_key, output_key_expr in mapping.outputs.items():
            value = _resolve_output_path(output_dict, output_key_expr)
            _set_dotted_key(out, dataset_key, value)
        return out

    for key, value in output_dict.items():
        out[key] = value
    return out


class StructuredTaskRunner:
    """High-level runner for sample-wise structured inference."""

    def __init__(self, compiled_task: CompiledStructuredTask, backend: StructuredOutputBackend):
        self.compiled_task = compiled_task
        self.backend = backend

    def run_sample(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        inputs = extract_inputs(sample, self.compiled_task.ast.dataset_mapping)
        raw_output = self.backend.infer(self.compiled_task.backend_program, inputs)
        normalized = normalize_backend_output(raw_output, self.compiled_task.ast.output_schema)
        validated = validate_output(self.compiled_task.output_model, normalized)
        return map_outputs_to_dataset(sample, validated, self.compiled_task.ast.dataset_mapping)


def build_pydantic_model(schema: SchemaAST, model_name: str) -> Type[BaseModel]:
    """Build a Pydantic model class from MMIRAGE schema AST."""

    fields: Dict[str, Tuple[Any, Any]] = {}
    for name, field in schema.fields.items():
        fields[name] = (_to_python_type(field.type_spec, nested_name=f"{model_name}_{name}"), ...)
    return create_model(model_name, **fields)


def _parse_schema(schema_data: Any, section: str) -> SchemaAST:
    if not isinstance(schema_data, dict):
        raise ValueError(f"{section} must be an object.")

    fields_source: Dict[str, Any]
    if schema_data.get("type") == "object":
        properties = schema_data.get("properties")
        if not isinstance(properties, dict) or not properties:
            raise ValueError(f"{section}.properties must be a non-empty object.")
        fields_source = properties
    else:
        fields_source = schema_data

    fields: Dict[str, FieldSpecAST] = {}
    for field_name, raw_type in fields_source.items():
        if not isinstance(field_name, str) or not field_name:
            raise ValueError(f"Invalid field name in {section}: {field_name!r}")

        if isinstance(raw_type, str):
            type_spec = _parse_type_spec(raw_type, f"{section}.{field_name}", nesting=0)
            description = ""
        elif isinstance(raw_type, dict):
            type_spec = _parse_type_spec(raw_type, f"{section}.{field_name}", nesting=0)
            description = str(raw_type.get("description", ""))
        else:
            raise ValueError(
                f"Invalid type declaration for {section}.{field_name}: {raw_type!r}"
            )

        fields[field_name] = FieldSpecAST(
            name=field_name,
            type_spec=type_spec,
            description=description,
        )

    if not fields:
        raise ValueError(f"{section} cannot be empty.")

    return SchemaAST(fields=fields)


def _parse_type_spec(raw_type: Any, path: str, nesting: int) -> TypeSpecAST:
    if isinstance(raw_type, str):
        raw_type = {"type": raw_type}

    if not isinstance(raw_type, dict):
        raise ValueError(f"{path} must be a string type or object.")

    type_name = raw_type.get("type")
    valid_types = {"string", "integer", "number", "boolean", "array", "object"}
    if type_name not in valid_types:
        raise ValueError(f"Unsupported type at {path}: {type_name!r}")

    if type_name in {"string", "integer", "number", "boolean"}:
        return TypeSpecAST(kind=type_name)

    if type_name == "array":
        items = raw_type.get("items")
        if items is None:
            raise ValueError(f"Array type at {path} requires 'items'.")
        item_type = _parse_type_spec(items, f"{path}.items", nesting=nesting)
        if item_type.kind not in {"string", "integer", "number", "boolean"}:
            raise ValueError(
                f"{path}.items must be a scalar type. Got: {item_type.kind}"
            )
        return TypeSpecAST(kind="array", items_kind=item_type.kind)

    # object
    properties = raw_type.get("properties")
    if not isinstance(properties, dict) or not properties:
        raise ValueError(f"Object type at {path} requires non-empty 'properties'.")
    if nesting >= 1:
        raise ValueError(f"Object nesting deeper than one level is not supported: {path}")

    parsed_props: Dict[str, TypeSpecAST] = {}
    for prop_name, prop_type in properties.items():
        prop_spec = _parse_type_spec(prop_type, f"{path}.properties.{prop_name}", nesting=nesting + 1)
        if prop_spec.kind == "object":
            raise ValueError(
                f"Nested object at {path}.properties.{prop_name} is not supported."
            )
        parsed_props[prop_name] = prop_spec

    return TypeSpecAST(kind="object", properties=parsed_props)


def _parse_dataset_mapping(
    data: Any,
    input_schema: SchemaAST,
    output_schema: SchemaAST,
) -> DatasetMappingAST:
    if not isinstance(data, dict):
        raise ValueError("dataset_mapping must be an object.")

    raw_inputs = data.get("inputs")
    raw_outputs = data.get("outputs")

    if not isinstance(raw_inputs, dict) or not raw_inputs:
        raise ValueError("dataset_mapping.inputs must be a non-empty object.")
    if raw_outputs is None:
        raw_outputs = {}
    if not isinstance(raw_outputs, dict):
        raise ValueError("dataset_mapping.outputs must be an object if provided.")

    inputs: Dict[str, str] = {}
    for input_name, expr in raw_inputs.items():
        if input_name not in input_schema.fields:
            raise ValueError(
                f"dataset_mapping.inputs references unknown input field '{input_name}'."
            )
        if not isinstance(expr, str) or not expr:
            raise ValueError(
                f"dataset_mapping.inputs['{input_name}'] must be a non-empty JMESPath string."
            )
        inputs[input_name] = expr

    outputs: Dict[str, str] = {}
    for dataset_key, output_key_expr in raw_outputs.items():
        if not isinstance(dataset_key, str) or not dataset_key:
            raise ValueError("dataset_mapping.outputs keys must be non-empty strings.")
        if not isinstance(output_key_expr, str) or not output_key_expr:
            raise ValueError(
                f"dataset_mapping.outputs['{dataset_key}'] must be a non-empty output path string."
            )

        root_output_key = output_key_expr.split(".")[0]
        if root_output_key not in output_schema.fields:
            raise ValueError(
                f"dataset_mapping.outputs['{dataset_key}'] references unknown output field '{root_output_key}'."
            )

        outputs[dataset_key] = output_key_expr

    return DatasetMappingAST(inputs=inputs, outputs=outputs)


def _parse_execution(data: Any) -> ExecutionAST:
    if not isinstance(data, dict):
        raise ValueError("execution must be an object.")

    backend = str(data.get("backend", "dspy")).strip() or "dspy"
    instruction = str(data.get("instruction", "")).strip()

    model = data.get("model")
    if model is not None and not isinstance(model, str):
        raise ValueError("execution.model must be a string if provided.")

    temperature = data.get("temperature")
    if temperature is not None:
        try:
            temperature = float(temperature)
        except (TypeError, ValueError) as exc:
            raise ValueError("execution.temperature must be numeric.") from exc

    max_tokens = data.get("max_tokens")
    if max_tokens is not None:
        try:
            max_tokens = int(max_tokens)
        except (TypeError, ValueError) as exc:
            raise ValueError("execution.max_tokens must be an integer.") from exc

    return ExecutionAST(
        backend=backend,
        instruction=instruction,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _to_python_type(type_spec: TypeSpecAST, nested_name: str = "Nested") -> Any:
    if type_spec.kind == "string":
        return str
    if type_spec.kind == "integer":
        return int
    if type_spec.kind == "number":
        return float
    if type_spec.kind == "boolean":
        return bool
    if type_spec.kind == "array":
        if type_spec.items_kind is None:
            return List[Any]
        return List[_to_python_type(TypeSpecAST(kind=type_spec.items_kind), nested_name)]
    if type_spec.kind == "object":
        if not type_spec.properties:
            return Dict[str, Any]
        nested_fields: Dict[str, Tuple[Any, Any]] = {}
        for key, prop_spec in type_spec.properties.items():
            nested_fields[key] = (_to_python_type(prop_spec, nested_name=f"{nested_name}_{key}"), ...)
        return create_model(nested_name, **nested_fields)
    return Any


def _normalize_value(value: Any, type_spec: TypeSpecAST) -> Any:
    if value is None:
        return value

    if type_spec.kind == "array" and isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    if type_spec.kind == "object" and isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return value


def _resolve_output_path(output_dict: Mapping[str, Any], expr: str) -> Any:
    current: Any = output_dict
    for token in expr.split("."):
        if not isinstance(current, Mapping) or token not in current:
            raise ValueError(f"Output path '{expr}' not found in validated output.")
        current = current[token]
    return current


def _set_dotted_key(target: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    if "." not in dotted_key:
        target[dotted_key] = value
        return

    keys = dotted_key.split(".")
    cur: MutableMapping[str, Any] = target
    for key in keys[:-1]:
        existing = cur.get(key)
        if existing is None or not isinstance(existing, MutableMapping):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


__all__ = [
    "CompiledStructuredTask",
    "DatasetMappingAST",
    "DspyBackend",
    "DspyCompiledProgram",
    "ExecutionAST",
    "FieldSpecAST",
    "SchemaAST",
    "StructuredOutputBackend",
    "StructuredTaskRunner",
    "TaskSpecAST",
    "TypeSpecAST",
    "build_pydantic_model",
    "compile_task",
    "extract_inputs",
    "map_outputs_to_dataset",
    "normalize_backend_output",
    "parse_task_spec",
    "parse_task_spec_yaml",
    "validate_output",
]
