import os
import sys
import json
import argparse
from typing import Dict, Any, List, Literal, Optional, Tuple

import yaml
import sglang as sgl
from dacite import from_dict
from dataclasses import asdict, dataclass, field
from datasets import load_from_disk, concatenate_datasets
from jmespath import search  # TODO: use compile to go faster
from pydantic import BaseModel, create_model
from transformers import GenerationConfig

from prompts import ASSISTANT_ONLY_MD_PROMPT


@dataclass
class EngineConfig:
    model_path: str
    tp_size: int = 4
    trust_remote_code: bool = True


@dataclass
class ProcessingGenParams:
    datasets: List[str]  # One or more paths to HF datasets saved with 'save_to_disk'
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
    
    def get_output_schema(self) -> Optional[BaseModel]:
        if self.output_type == "JSON" and self.output_schema:
            fields = {var: (str, ...) for var in self.output_schema} # ... means required field
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
    output_schema: OutputSchema


@dataclass
class MirageConfig:
    engine: EngineConfig
    sampling_params: Dict[str, Any] | GenerationConfig
    processing_gen_params: ProcessingGenParams
    processing_params: ProcessingParams

    def __post_init__(self):
        self.sampling_params = GenerationConfig(**self.sampling_params)


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
        cfg: dict = yaml.safe_load(f) or {}

    def expand_env_vars(obj):
        if isinstance(obj, dict):
            return {key: expand_env_vars(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [expand_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            return os.path.expandvars(obj)
        else:
            return obj

    cfg = expand_env_vars(cfg)
    cfg_obj = from_dict(MirageConfig, cfg)
    engine_args = cfg_obj.engine
    llm = sgl.Engine(**asdict(engine_args))

    return llm, cfg_obj


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
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=384,
        help="Override max_new_tokens in sampling params.",
    )
    args = ap.parse_args()

    # -------------------------
    # Load SGLang engine + sampling + batch size
    # -------------------------
    llm, cfg = load_engine_from_yaml(args.config)
    sampling_params = cfg.sampling_params
    assert isinstance(sampling_params, GenerationConfig)
    processing_gen_params = cfg.processing_gen_params
    processing_params = cfg.processing_params

    datasets = processing_gen_params.datasets
    if not datasets:
        raise ValueError(
            "No datasets provided in config.processing_gen_params.datasets"
        )
    shard_id = processing_gen_params.shard_id
    num_shards = processing_gen_params.num_shards

    if not (0 <= shard_id < num_shards):
        raise ValueError(f"Invalid shard_id={shard_id}, num_shards={num_shards}")

    os.makedirs(processing_gen_params.output_dir, exist_ok=True)
    shard_out_dir = os.path.join(processing_gen_params.output_dir, f"shard_{shard_id}")
    os.makedirs(shard_out_dir, exist_ok=True)

    # -------------------------
    # Load all input datasets and concatenate
    # -------------------------
    ds_list = [load_from_disk(p) for p in datasets]
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

    conv_field = processing_gen_params.conversations_field
    if conv_field not in ds_shard.column_names:
        raise ValueError(
            f"Expected conversations column '{conv_field}', "
            f"but dataset has columns: {ds_shard.column_names}"
        )

    # Apply script-level override for max_new_tokens
    if args.max_new_tokens is not None:
        sampling_params.max_new_tokens = int(args.max_new_tokens)
    
    sampling_params: Dict[str, Any] = sampling_params.to_dict()

    # -------------------------
    # Batched rewrite function for HF map
    # -------------------------
    def extract_input_from_conv(conv: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract value from conversation using processing_params.inputs."""
        
        input_vars: Dict[str, Any] = {}
        for input_var in processing_params.inputs:
            value = search(input_var.key, conv)
            input_vars[input_var.name] = value
        return input_vars
    
    def rewrite_batch(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        conv_batch = batch[conv_field]

        prompts: List[Tuple[int, OutputVar, str]] = [] # (example_idx, output_var, prompt_str)
        vars: List[Dict[str, Any]] = [] # input vars for each example
        locs: List[Tuple[int, int]] = []  # (example_idx, assistant_turn_idx)

        # First pass: collect prompts where there is a non-empty assistant turn
        for i, conv in enumerate(conv_batch):
            if not isinstance(conv, list) or not conv:
                continue
            
            current_vars = extract_input_from_conv(conv)
            vars.append(current_vars)

        # Nothing to rewrite in this batch
        if not prompts:
            return {conv_field: conv_batch}

        try:
            # Non-streaming synchronous batch generation
            outputs: List[Dict[str, Any]] = []
            for output in processing_params.outputs:
                prompts_for_output = [output.prompt.format(**var) for var in vars]
                sampling_params_output = sampling_params.copy()
                if output.output_type == "JSON":
                    json_schema = output.get_output_schema()
                    if json_schema is None:
                        raise ValueError(
                            f"Output variable {output.name} has output_type=JSON "
                            "but no output_schema defined."
                        )

                    sampling_params_output["json_schema"] = json.dumps(json_schema)

                outputs += llm.generate(prompts_for_output, sampling_params_output)
        except Exception as e:
            print(
                f"[shard {shard_id}] Batch generation failed: {e}",
                file=sys.stderr,
            )
            # On error, keep original conversations for this batch
            return {conv_field: conv_batch}

        if not isinstance(outputs, list) or len(outputs) != len(prompts):
            print(
                f"[shard {shard_id}] Unexpected outputs length from llm.generate: "
                f"expected {len(prompts)}, got {len(outputs) if isinstance(outputs, list) else 'non-list'}",
                file=sys.stderr,
            )
            return {conv_field: conv_batch}
        
        new_conv_batch = []
        for i, ((ex_idx, output_var, _), output) in enumerate(zip(prompts, outputs)):
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
        # Copy original conversations and fill in rewritten Markdown
        new_conv_batch = list(conv_batch)
        for out_idx, (ex_idx, turn_idx) in enumerate(locs):
            out = outputs[out_idx]
            md_text = out.get("text", "")

            orig_conv = new_conv_batch[ex_idx]
            # create shallow copies so we don't mutate shared objects
            conv_list = list(orig_conv)
            turn = dict(conv_list[turn_idx])
            turn["content"] = md_text
            conv_list[turn_idx] = turn
            new_conv_batch[ex_idx] = conv_list
        """

        # Only return the updated conversations column; HF keeps other columns
        return {conv_field: new_conv_batch}

    # -------------------------
    # Apply map with batching
    # -------------------------
    ds_processed = ds_shard.map(
        rewrite_batch,
        batched=True,
        batch_size=processing_gen_params.batch_size,
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
