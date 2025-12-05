import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Tuple, TypeAlias, Union, cast

import sglang as sgl
import yaml
from dacite import from_dict
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from jmespath import search  # TODO: use compile to go faster
from prompts import ASSISTANT_ONLY_MD_PROMPT
from pydantic import create_model
from transformers import GenerationConfig

EnvValue: TypeAlias = Union[str, List["EnvValue"], Dict[str, "EnvValue"]]


@dataclass
class EngineConfig:
    model_path: str
    tp_size: int = 4
    trust_remote_code: bool = True


@dataclass
class DatasetConfig:
    path: str
    type: Literal["JSONL", "loadable"] = "loadable"


@dataclass
class ProcessingGenParams:
    datasets: List[
        DatasetConfig
    ]  # One or more paths to HF datasets saved with 'save_to_disk'
    output_dir: str  # Root directory for shard outputs
    num_shards: int | str = (
        1  # Total number of shards (matches your sbatch array size).
    )
    shard_id: int | str = (
        0  # Index of this shard (0-based; usually $SLURM_ARRAY_TASK_ID).
    )
    conversations_field: str = (
        "conversations"  # Name of the column containing the list of dialog turns.
    )
    batch_size: int | str = 64  # Batch size for processing

    def __post_init__(self):
        if isinstance(self.num_shards, str):
            self.num_shards = int(self.num_shards) if self.num_shards.isdigit() else 1
        if isinstance(self.shard_id, str):
            self.shard_id = int(self.shard_id) if self.shard_id.isdigit() else 0
        if isinstance(self.batch_size, str):
            self.batch_size = int(self.batch_size) if self.batch_size.isdigit() else 64
        self.batch_size = max(self.batch_size, 1)

    def get_num_shards(self) -> int:
        return cast(int, self.num_shards)

    def get_shard_id(self) -> int:
        return cast(int, self.shard_id)

    def get_batch_size(self) -> int:
        return cast(int, self.batch_size)


@dataclass
class InputVar:
    name: str
    key: str


@dataclass
class OutputVar:
    name: str
    type: str
    output_type: Literal["plain", "JSON"]
    prompt: str
    output_schema: List[str] = field(
        default_factory=list
    )  # empty list if output_type is "plain"

    def get_output_schema(self):
        if self.output_type == "JSON" and self.output_schema:
            fields: Dict[str, Any] = {
                var: (str, ...) for var in self.output_schema
            }  # ... means required field
            return create_model(f"OutputSchema", **fields)
        return None


@dataclass
class Message:
    role: str
    content: str


@dataclass
class OutputSchema:
    conversations: List[Message]
    modalities: str


@dataclass
class ProcessingParams:
    inputs: List[InputVar]
    outputs: List[OutputVar]
    output_schema: Dict[str, Any]


@dataclass
class MirageConfig:
    engine: EngineConfig
    sampling_params: Dict[str, Any]
    processing_gen_params: ProcessingGenParams
    processing_params: ProcessingParams

    def __post_init__(self):
        self.sampling_params_internal = GenerationConfig(
            **cast(Dict[str, Any], self.sampling_params)
        )

    def get_sampling_params(self) -> GenerationConfig:
        return self.sampling_params_internal


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
        - "/path/to/dataset1"
        - "/path/to/dataset2"
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
            {{ assistant_answer }}
          output_schema:
            question: question_variable
            explanation: explanation_variable
            answer: answer_variable

      output_schema:
        conversations:
        - role: user
          content: {{ user_prompt }}
        - role: assistant
          content: {{ formatted_answer }}
        modalities: {{ modalities }}
    """
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
    def extract_template_vars(obj: Any) -> set:
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


def build_prompt(text: str) -> str:
    """Build the Markdown-enhancement prompt for a single assistant message."""
    payload = json.dumps({"assistant_text": text}, ensure_ascii=False)
    return ASSISTANT_ONLY_MD_PROMPT.format(payload=payload)


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser(
        "Rewrite the assistant turn inside `conversations` into Markdown using SGLang + HF map + sharding."
    )
    ap.add_argument(
        "--config",
        default="config-sglang.yaml",
        help="YAML config for SGLang engine + sampling + batch_size.",
    )
    args = ap.parse_args()

    # -------------------------
    # Load SGLang engine + sampling + batch size
    # -------------------------
    llm, cfg = load_engine_from_yaml(args.config)
    validate_processing_params(cfg.processing_params)
    sampling_params = cfg.get_sampling_params()
    assert isinstance(sampling_params, GenerationConfig)
    processing_gen_params = cfg.processing_gen_params
    processing_params = cfg.processing_params

    datasets = processing_gen_params.datasets
    if not datasets:
        raise ValueError(
            "No datasets provided in config.processing_gen_params.datasets"
        )
    shard_id = processing_gen_params.get_shard_id()
    num_shards = processing_gen_params.get_num_shards()

    if not (0 <= shard_id < num_shards):
        raise ValueError(f"Invalid shard_id={shard_id}, num_shards={num_shards}")

    os.makedirs(processing_gen_params.output_dir, exist_ok=True)
    shard_out_dir = os.path.join(processing_gen_params.output_dir, f"shard_{shard_id}")
    os.makedirs(shard_out_dir, exist_ok=True)

    # -------------------------
    # Load all input datasets and concatenate
    # -------------------------
    def load_datasets_from_configs(configs: List[DatasetConfig]) -> List[Dataset]:
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
                    assert not isinstance(ds, IterableDatasetDict)
                    assert not isinstance(ds, IterableDataset)
                else:
                    ds = load_from_disk(path)

                if isinstance(ds, DatasetDict):
                    # Merge all splits into one Dataset
                    ds = concatenate_datasets([ds[split] for split in ds.keys()])

                valid_ds.append(ds)
            except Exception as e:
                print(f"⚠️ Failed to load dataset from {path}, skipping. Reason: {e}")
        return valid_ds

    ds_list = load_datasets_from_configs(datasets)
    if len(ds_list) == 1:
        ds_all = ds_list[0]
    else:
        ds_all = concatenate_datasets(ds_list)

    total_rows = len(ds_all)

    ds_shard = ds_all.shard(num_shards=num_shards, index=shard_id)
    shard_rows = len(ds_shard)

    print(
        f"Loaded {len(datasets)} dataset(s): {datasets} "
        f"→ {total_rows} total rows; this shard has {shard_rows} rows."
    )

    sampling_params_dict: Dict[str, Any] = sampling_params.to_dict()

    # -------------------------
    # Batched rewrite function for HF map
    # -------------------------

    def extract_input_vars(sample: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively extract all input variables from a dataset sample."""
        input_vars: Dict[str, Any] = {}
        for input_var in processing_params.inputs:
            value = search(input_var.key, sample)
            input_vars[input_var.name] = value
        return input_vars

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

    def rewrite_batch(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        prompts: List[
            Tuple[int, OutputVar, str]
        ] = []  # (example_idx, output_var, prompt_str)
        vars: List[Dict[str, Any]] = []  # input vars for each example

        # turn the dictionary of lists into a list of dictionaries
        batch_list: List[Dict[str, Any]] = []
        for i, (key, values) in enumerate(batch.items()):
            if i == 0:  # first column
                batch_list += [{key: x} for x in values]
            else:
                assert len(values) == len(batch_list)
                for j, x in enumerate(values):
                    batch_list[j][key] = x

        for sample in batch_list:
            current_vars = extract_input_vars(sample)
            vars.append(current_vars)

        try:
            # Non-streaming synchronous batch generation
            outputs: List[Dict[str, Any]] = []
            for output in processing_params.outputs:
                prompts_for_output = [output.prompt.format(**var) for var in vars]
                prompts += [(i, output, x) for i, x in enumerate(prompts_for_output)]

                sampling_params_output = sampling_params_dict.copy()
                if output.output_type == "JSON":
                    json_schema = output.get_output_schema()
                    if json_schema is None:
                        raise ValueError(
                            f"Output variable {output.name} has output_type=JSON "
                            "but no output_schema defined."
                        )

                    sampling_params_output["json_schema"] = json.dumps(json_schema)

                outputs_for_output = llm.generate(
                    prompts_for_output, sampling_params_output
                )
                assert len(prompts_for_output) == len(outputs_for_output)

                outputs += outputs_for_output
        except Exception as e:
            print(
                f"[shard {shard_id}] Batch generation failed: {e}",
                file=sys.stderr,
            )
            # On error, keep original conversations for this batch
            return batch

        if not isinstance(outputs, list) or len(outputs) != len(prompts):
            print(
                f"[shard {shard_id}] Unexpected outputs length from llm.generate: "
                f"expected {len(prompts)}, got {len(outputs) if isinstance(outputs, list) else 'non-list'}",
                file=sys.stderr,
            )
            return batch

        new_results = []
        for (ex_idx, output_var, _), output in zip(prompts, outputs):
            out_text = output.get("text", "").strip()
            vars_ex = vars[ex_idx]
            vars_ex[output_var.name] = out_text

            # Rebuild the output according to output_schema template
            output_schema = processing_params.output_schema
            filled_output = fill_template_recursive(output_schema, vars_ex)

            new_results.append(filled_output)

        """
        new_conv_batch = []
        for (ex_idx, output_var, _), output in zip(prompts, outputs):
            out_text = output.get("text", "").strip()
            vars_ex = vars[ex_idx]
            vars_ex[output_var.name] = out_text

            # Rebuild the full conversation according to output_schema
            output_schema = processing_params.output_schema
            # Start with original conversation
            orig_conv = conv_batch[ex_idx]
            conv_list = list(orig_conv)  # shallow copy
            # Rebuild according to output_schema
            new_conv: List[Dict[str, Any]] = []
            for msg_schema in output_schema.conversations:
                role = msg_schema.role
                content_template = msg_schema.content
                content_filled = content_template.format(**vars_ex)
                new_conv.append({"role": role, "content": content_filled})

            # Replace conversation in conv_list
            conv_list = new_conv
            # Add other fields if needed (e.g., modalities)
            if hasattr(output_schema, "modalities"):
                modalities_filled = output_schema.modalities.format(**vars_ex)
                # Assuming modalities is a top-level field in the dataset
                conv_list.append({"modalities": modalities_filled})

            new_conv_batch.append(conv_list)
        """

        # Only return the updated conversations column; HF keeps other columns

        # Build result dict with all columns from output_schema
        result_batch: Dict[str, List[Any]] = {}
        for key in processing_params.output_schema.keys():
            result_batch[key] = [result.get(key) for result in new_results]

        return result_batch

    # -------------------------
    # Apply map with batching
    # -------------------------
    ds_processed = ds_shard.map(
        rewrite_batch,
        batched=True,
        batch_size=processing_gen_params.get_batch_size(),
        load_from_cache_file=False,
        desc=f"Shard {shard_id}/{num_shards - 1}",
    )

    # -------------------------
    # Save shard as its own HF dataset (all columns preserved)
    # -------------------------
    ds_processed.save_to_disk(shard_out_dir)

    try:
        llm.shutdown()
    except Exception:
        pass

    print(
        f"✅ shard_id={shard_id} num_shards={num_shards} "
        f"total_rows={total_rows} shard_rows={shard_rows} "
        f"out_dir={shard_out_dir}"
    )


if __name__ == "__main__":
    main()
