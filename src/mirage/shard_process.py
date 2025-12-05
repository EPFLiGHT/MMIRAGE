import argparse
import json
import os
import sglang as sgl
import sys
from typing import Any, Dict, List, Tuple

from config import InputVar, OutputVar
from utils import extract_input_vars, fill_template_recursive, load_datasets_from_configs, load_engine_from_yaml, validate_processing_params

def rewrite_batch(batch: Dict[str, List[Any]], processing_inputs: List[InputVar], processing_outputs: List[OutputVar], sampling_params: Dict[str, Any], output_schema: Dict[str, Any], llm: sgl.Engine, shard_id: int) -> Dict[str, List[Any]]:
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
        current_vars = extract_input_vars(processing_inputs, sample)
        vars.append(current_vars)

    try:
        # Non-streaming synchronous batch generation
        outputs: List[Dict[str, Any]] = []
        for output in processing_outputs:
            prompts_for_output = [output.prompt.format(**var) for var in vars]
            prompts += [(i, output, x) for i, x in enumerate(prompts_for_output)]

            sampling_params_output = sampling_params.copy()
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
        filled_output = fill_template_recursive(output_schema, vars_ex)

        new_results.append(filled_output)

    # Only return the updated conversations column; HF keeps other columns

    # Build result dict with all columns from output_schema
    result_batch: Dict[str, List[Any]] = {}
    for key in output_schema.keys():
        result_batch[key] = [result.get(key) for result in new_results]

    return result_batch

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
    sampling_params = cfg.sampling_params
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
    ds_all = load_datasets_from_configs(datasets)
    total_rows = len(ds_all)

    ds_shard = ds_all.shard(num_shards=num_shards, index=shard_id)
    shard_rows = len(ds_shard)

    print(
        f"Loaded {len(datasets)} dataset(s): {datasets} "
        f"→ {total_rows} total rows; this shard has {shard_rows} rows."
    )

    # -------------------------
    # Apply map with batching
    # -------------------------
    ds_processed = ds_shard.map(
        rewrite_batch,
        batched=True,
        batch_size=processing_gen_params.get_batch_size(),
        load_from_cache_file=False,
        desc=f"Shard {shard_id}/{num_shards - 1}",
        fn_kwargs={"shard_id": shard_id, "llm": llm, "processing_outputs": processing_params.outputs, "processing_inputs": processing_params.inputs, "sampling_params": sampling_params, "output_schema": processing_params.output_schema},
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
