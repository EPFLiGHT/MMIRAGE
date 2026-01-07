import argparse
import os
import sys
from typing import Any, Dict, List, Tuple
from sglang.srt.parser.conversation import chat_templates
import sglang as sgl

from mirage.config import InputVar, OutputVar
from mirage.utils import (
    extract_input_vars,
    fill_template_recursive,
    load_datasets_from_configs,
    load_engine_from_yaml,
    validate_processing_params,
)


def build_multimodal_prompt(
    prompt_text: str, vars_dict: Dict[str, Any], processing_inputs: List[InputVar]
) -> Tuple[str, List[Any]]:
    """Build a prompt and extract images for SGLang Engine."""
    formatted_prompt = prompt_text.format(**vars_dict)

    images = []
    for inp in processing_inputs:
        if inp.is_image():
            image_value = vars_dict.get(inp.name)
            if image_value is not None:
                images.append(image_value)

    return formatted_prompt, images


def rewrite_batch(
    batch: Dict[str, List[Any]],
    processing_inputs: List[InputVar],
    processing_outputs: List[OutputVar],
    sampling_params: Dict[str, Any],
    output_schema: Dict[str, Any],
    llm: sgl.Engine,
    shard_id: int,
    chat_template: str,
) -> Dict[str, List[Any]]:
    vars_samples: List[Dict[str, Any]] = []  # input vars for each example

    # turn the dictionary of lists into a list of dictionaries
    batch_size = len(next(iter(batch.values())))
    batch_list: List[Dict[str, Any]] = [
        {k: batch[k][i] for k in batch.keys()} for i in range(batch_size)
    ]
    nb_samples = len(batch_list)

    for sample in batch_list:
        current_vars = extract_input_vars(processing_inputs, sample)
        vars_samples.append(current_vars)

    try:
        # Generate and fill vars_samples[i][output.name]
        for output in processing_outputs:
            # Build prompts and extract images
            prompt_image_pairs = [
                build_multimodal_prompt(output.prompt, var, processing_inputs)
                for var in vars_samples
            ]
            prompts_for_output = [pair[0] for pair in prompt_image_pairs]
            images_for_output = [pair[1] for pair in prompt_image_pairs]

            sampling_params_output = dict(sampling_params)
            if output.output_type == "JSON":
                json_schema = output.get_output_schema()
                if json_schema is None:
                    raise ValueError(
                        f"Output variable {output.name} has output_type=JSON "
                        "but no output_schema defined."
                    )
                sampling_params_output["json_schema"] = json_schema.model_json_schema()

            # Separate samples into text-only and multimodal groups
            text_only_indices = []
            multimodal_indices = []
            
            for i in range(nb_samples):
                if len(images_for_output[i]) > 0:
                    multimodal_indices.append(i)
                else:
                    text_only_indices.append(i)
            
            # Process text-only samples in batch if any exist
            if text_only_indices:
                text_only_prompts = [prompts_for_output[i] for i in text_only_indices]
                
                try:
                    text_only_outputs = llm.generate(
                        prompt=text_only_prompts,
                        sampling_params=sampling_params_output,
                    )
                    
                    if not isinstance(text_only_outputs, list) or len(text_only_outputs) != len(text_only_indices):
                        raise RuntimeError(
                            f"Mismatch between text-only prompts and outputs for '{output.name}': "
                            f"{len(text_only_prompts)} vs "
                            f"{len(text_only_outputs) if isinstance(text_only_outputs, list) else 'non-list'}"
                        )
                    
                    for idx, i in enumerate(text_only_indices):
                        vars_samples[i][output.name] = text_only_outputs[idx].get("text", "").strip()
                
                except Exception as e:
                    print(
                        f"[shard {shard_id}] Batch generation failed for text-only samples in output '{output.name}': {e}",
                        file=sys.stderr,
                        flush=True,
                    )
                    # On error, set empty strings for failed samples
                    for i in text_only_indices:
                        vars_samples[i][output.name] = ""
            
            # Process multimodal samples in batch if any exist
            if multimodal_indices:
                # Validate chat template exists and extract image token once
                if chat_template not in chat_templates:
                    raise ValueError(
                        f"Chat template '{chat_template}' not found. "
                        f"Available templates: {list(chat_templates.keys())}"
                    )
                
                conv = chat_templates[chat_template].copy()
                image_token = conv.image_token
                
                # Prepare batched inputs for multimodal generation
                multimodal_prompts = [
                    prompts_for_output[i] + f"\n{image_token}\n" 
                    for i in multimodal_indices
                ]
                multimodal_images = [images_for_output[i] for i in multimodal_indices]
                
                try:
                    multimodal_outputs = llm.generate(
                        prompt=multimodal_prompts,
                        sampling_params=sampling_params_output,
                        image_data=multimodal_images,
                    )
                    
                    if not isinstance(multimodal_outputs, list) or len(multimodal_outputs) != len(multimodal_indices):
                        raise RuntimeError(
                            f"Mismatch between multimodal prompts and outputs for '{output.name}': "
                            f"{len(multimodal_prompts)} vs "
                            f"{len(multimodal_outputs) if isinstance(multimodal_outputs, list) else 'non-list'}"
                        )
                    
                    for idx, i in enumerate(multimodal_indices):
                        vars_samples[i][output.name] = multimodal_outputs[idx].get("text", "").strip()
                
                except Exception as e:
                    print(
                        f"[shard {shard_id}] Batch generation failed for multimodal samples in output '{output.name}': {e}",
                        file=sys.stderr,
                        flush=True,
                    )
                    # On error, set empty strings for failed samples
                    for i in multimodal_indices:
                        vars_samples[i][output.name] = ""

    except Exception as e:
        print(
            f"[shard {shard_id}] Batch generation failed: {e}",
            file=sys.stderr,
            flush=True,
        )
        # On error, keep original batch
        return batch

    # Rebuild the output according to output_schema template
    new_results = []
    for vars_sample in vars_samples:
        filled_output = fill_template_recursive(output_schema, vars_sample)
        new_results.append(filled_output)

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
    
    # Get chat template from config
    chat_template = cfg.engine.chat_template
    
    # Validate chat template early to provide clear error feedback
    if chat_template not in chat_templates:
        available = list(chat_templates.keys())
        raise ValueError(
            f"Chat template '{chat_template}' not found. "
            f"Available templates: {available}. "
            f"Please set a valid 'chat_template' in your engine config. "
            f"Common template: 'qwen2-vl'"
        )
    
    print(f"Using chat template: {chat_template}")

    datasets = processing_gen_params.datasets
    if not datasets:
        raise ValueError("No datasets provided in config.processing_gen_params.datasets")

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

    # Handle empty shards gracefully
    if shard_rows == 0:
        print(f"⚠️  Shard {shard_id} is empty (dataset has only {total_rows} samples for {num_shards} shards).")
        print(f"✅ Saving empty shard to {shard_out_dir}")
        ds_shard.save_to_disk(shard_out_dir)
        try:
            llm.shutdown()
        except Exception:
            pass
        return

    # -------------------------
    # Apply map with batching
    # -------------------------
    ds_processed = ds_shard.map(
        rewrite_batch,
        batched=True,
        batch_size=processing_gen_params.get_batch_size(),
        load_from_cache_file=False,
        desc=f"Shard {shard_id}/{num_shards - 1}",
        fn_kwargs={
            "shard_id": shard_id,
            "llm": llm,
            "processing_outputs": processing_params.outputs,
            "processing_inputs": processing_params.inputs,
            "sampling_params": sampling_params,
            "output_schema": processing_params.output_schema,
            "chat_template": chat_template,
        },
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
