from __future__ import annotations

import os
import re
import sglang as sgl
import yaml

from dacite import from_dict
from dataclasses import asdict
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, concatenate_datasets, load_dataset, load_from_disk
from jmespath import search
from typing import Any, Dict, List, Set, Tuple, TypeAlias, TYPE_CHECKING, Union, cast

EnvValue: TypeAlias = Union[str, List["EnvValue"], Dict[str, "EnvValue"]]

if TYPE_CHECKING:
    from mirage.config import DatasetConfig, InputVar, MirageConfig, ProcessingParams

# Utilities

# -------------------------
# helpers
# -------------------------
def load_engine_from_yaml(config_path: str) -> Tuple[sgl.Engine, MirageConfig]:
    """
    Load SGLang engine, sampling params, and batch size from YAML config.

    Example config:

    engine:
      model_path: "meta-llama/Meta-Llama-3.1-8B-Instruct"

    sampling_params:
      temperature: 0.2
      top_p: 0.9

    processing_gen_params:
      datasets:
        - path: "/path/to/dataset1"
          type: loadable
        - path: "/path/to/dataset2.jsonl"
          type: JSONL
      output_dir: "/path/to/output"
      num_shards: 8
      shard_id: 0
      conversations_field: "conversations"

    processing_params:
      inputs:
        - name: assistant_answer
          key: conversations[1].content
        - name: user_prompt
          key: conversations[0].content
        - name: modalities
          key: modalities

      outputs:
        - name: formatted_answer
          type: llm
          output_type: plain
          prompt: |
            Reformat the answer in a markdown format without adding anything else:
            {assistant_answer}
          output_schema:
            - question
            - explanation
            - answer

      output_schema:
        conversations:
        - role: user
          content: {user_prompt}
        - role: assistant
          content: {formatted_answer}
        modalities: {modalities}
    """
    from mirage.config import MirageConfig

    with open(config_path, "r") as f:
        cfg: EnvValue = yaml.safe_load(f) or {}

    def expand_env_vars(obj: EnvValue) -> EnvValue:
        if isinstance(obj, dict):
            return {key: expand_env_vars(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [expand_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            return os.path.expandvars(obj)
        else:
            return obj

    cfg = expand_env_vars(cfg)
    cfg_obj = from_dict(MirageConfig, cast(dict, cfg))
    engine_args = cfg_obj.engine
    llm = sgl.Engine(**asdict(engine_args))

    return llm, cfg_obj


def validate_processing_params(params: ProcessingParams) -> None:
    """
    Validate that ProcessingParams.output_schema uses only variables defined in
    inputs and outputs.

    Raises:
        ValueError: If undefined variables are found in output_schema or prompts
    """

    # Collect all defined variable names
    defined_vars = set()

    # From inputs
    for input_var in params.inputs:
        defined_vars.add(input_var.name)

    # From outputs
    for output_var in params.outputs:
        defined_vars.add(output_var.name)

    # Extract all template variables (patterns like {var_name})
    def extract_template_vars(obj: Any) -> Set[str]:
        """Recursively extract all {var} patterns from templates."""
        vars_found = set()

        if isinstance(obj, str):
            # Find all {variable_name} patterns
            matches = re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", obj)
            vars_found.update(matches)
        elif isinstance(obj, dict):
            for value in obj.values():
                vars_found.update(extract_template_vars(value))
        elif isinstance(obj, list):
            for item in obj:
                vars_found.update(extract_template_vars(item))

        return vars_found

    # Check output_schema templates
    schema_vars = extract_template_vars(params.output_schema)

    # Check output prompts
    prompt_vars = set()
    for output_var in params.outputs:
        prompt_vars.update(extract_template_vars(output_var.prompt))

    # Validate all template variables are defined
    undefined_in_schema = schema_vars - defined_vars
    undefined_in_prompts = prompt_vars - defined_vars

    errors = []

    if undefined_in_schema:
        errors.append(
            f"Undefined variables in output_schema: {', '.join(sorted(undefined_in_schema))}. "
            f"Available variables: {', '.join(sorted(defined_vars))}"
        )

    if undefined_in_prompts:
        errors.append(
            f"Undefined variables in output prompts: {', '.join(sorted(undefined_in_prompts))}. "
            f"Available variables: {', '.join(sorted(defined_vars))}"
        )

    if errors:
        raise ValueError("\n".join(errors))

    print(
        f"✅ ProcessingParams validation passed. "
        f"Defined variables: {', '.join(sorted(defined_vars))}"
    )

def load_datasets_from_configs(configs: List[DatasetConfig]) -> Dataset:
    valid_ds = []
    for ds_config in configs:
        path = ds_config.path

        if not os.path.exists(path):
            print(f"⚠️ Dataset path does not exist, skipping: {path}")
            continue
        try:
            if ds_config.type == "JSONL":
                ds = load_dataset("json", data_files=path, streaming=False)
                # no support of iterable datasets
                if isinstance(ds, (IterableDatasetDict, IterableDataset)):
                    raise ValueError(f"Iterable datasets are not supported for path: {path}")
            else:
                ds = load_from_disk(path)

            if isinstance(ds, DatasetDict):
                # Merge all splits into one Dataset
                ds = concatenate_datasets([ds[split] for split in ds.keys()])

            valid_ds.append(ds)
        except Exception as e:
            print(f"⚠️ Failed to load dataset from {path}, skipping. Reason: {e}")

    if not valid_ds:
        raise RuntimeError("No valid datasets loaded from the provided configs.")
    elif len(valid_ds) == 1:
        return valid_ds[0]
    else:
        return concatenate_datasets(valid_ds)


def extract_input_vars(input_vars: List[InputVar], sample: Dict[str, Any]) -> Dict[str, Any]:
    """Extract input variables from a dataset sample using JMESPath queries."""

    ret: Dict[str, Any] = {}
    for input_var in input_vars:
        value = search(input_var.key, sample)
        if value is None:
            raise ValueError(
                f"Input variable '{input_var.name}' with key '{input_var.key}' "
                "not found in the sample."
            )
        ret[input_var.name] = value

    return ret

def fill_template_recursive(template: Any, vars_dict: Dict[str, Any]) -> Any:
    """Recursively fill templates in nested structures."""

    if isinstance(template, str):
        return template.format(**vars_dict)
    elif isinstance(template, dict):
        return {
            k: fill_template_recursive(v, vars_dict) for k, v in template.items()
        }
    elif isinstance(template, list):
        return [fill_template_recursive(item, vars_dict) for item in template]
    else:
        return template