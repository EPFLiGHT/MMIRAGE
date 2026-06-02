# 🖼️ Multimodal Processing

This page explains how to use MMIRAGE with vision-language models (VLMs) to
process datasets that include images.

Before reading this page, familiarise yourself with [Concepts](concepts.md)
and the [Quickstart](quickstart.md) text-only example.

---

## Overview

MMIRAGE supports multimodal inputs natively.
When a dataset sample includes an image reference, MMIRAGE resolves it to a
PIL Image object and passes it alongside the text prompt to the VLM.

The key additions compared to a text-only pipeline are:

- one or more `inputs` with `type: image`
- `image_base_path` in the dataset config to resolve relative paths
- a `chat_template` on the processor matching your VLM's expected format

---

## Dataset configuration for images

Add `image_base_path` to the dataset entry in `loading_params.datasets` to
specify the directory where image files are stored:

```yaml
loading_params:
  datasets:
    - path: /path/to/dataset
      type: loadable
      output_dir: /path/to/output/shards
      image_base_path: /path/to/images
```

If your dataset already stores absolute paths in the `image_path` field,
you can omit `image_base_path`.

---

## Image input variables

Declare image inputs in `processing_params.inputs` by setting `type: image`:

```yaml
processing_params:
  inputs:
    - name: image
      key: image_path
      type: image
    - name: question
      key: question
```

The `key` extracts a value from each sample (via JMESPath).
For image inputs, this value is treated as a file path.
MMIRAGE resolves it as follows:

1. If `image_base_path` is set on the dataset config, the path is joined with that prefix.
2. The resolved path is loaded as a PIL Image and stored under `name`.

Inside your prompt template, you can reference the image variable by name.
MMIRAGE places it in the correct position in the multimodal message:

```yaml
outputs:
  - name: answer
    type: llm
    output_type: plain
    prompt: |
      {{ image }}
      Answer this question about the image:
      {{ question }}
```

---

## Chat template

VLMs expect their inputs in a specific format.
Set `chat_template` on the processor to enable the correct message structure:

```yaml
processors:
  - type: llm
    server_args:
      model_path: Qwen/Qwen2-VL-7B-Instruct
      tp_size: 4
      trust_remote_code: true
    chat_template: qwen2-vl
    default_sampling_params:
      temperature: 0.1
      top_p: 0.95
      max_new_tokens: 768
```

The `chat_template` value is passed directly to the SGLang engine.
Common values:

| Model family | `chat_template` value |
|---|---|
| Qwen2-VL | `qwen2-vl` |
| LLaVA-style | `llava` |
| InternVL | `internvl` |

Leave `chat_template` empty (or omit it) for text-only LLMs.

---

## Complete multimodal example

```yaml
processors:
  - type: llm
    server_args:
      model_path: Qwen/Qwen2-VL-7B-Instruct
      tp_size: 4
      trust_remote_code: true
    chat_template: qwen2-vl
    default_sampling_params:
      temperature: 0.1
      top_p: 0.95
      max_new_tokens: 768

loading_params:
  state_dir: /path/to/state
  datasets:
    - path: /path/to/image_dataset
      type: loadable
      output_dir: /path/to/output/shards
      image_base_path: /path/to/images
  num_shards: 8
  shard_id: "$SLURM_ARRAY_TASK_ID"
  batch_size: 8

processing_params:
  inputs:
    - name: image
      key: image_path
      type: image
    - name: question
      key: question

  outputs:
    - name: answer
      type: llm
      output_type: plain
      prompt: |
        {{ image }}
        Answer this question about the image:
        {{ question }}

  output_schema:
    question: "{{ question }}"
    answer: "{{ answer }}"
    image_path: "{{ image }}"

execution_params:
  mode: slurm
  account: my_account
  job_name: vlm-pipeline
  nodes: 1
  ntasks_per_node: 1
  gpus: 4
  cpus_per_task: 32
  time_limit: "05:59:59"
  retry: true
  merge: true
  max_retries: 2
```

---

## Batch size for image workloads

VLMs process significantly fewer samples per second than text-only LLMs.
Typical guidance:

- Start with `batch_size: 8` and reduce if you hit OOM errors.
- Use `tp_size` matching the number of GPUs on your node.
- For large images, consider reducing `max_new_tokens` to save memory.

---

## See also

- [Quickstart](quickstart.md) — text-only example to compare with
- [Pipeline](pipeline.md) — how image inputs flow through the mapper
- [Configuration Reference](configuration.md) — full `loading_params` and `processors` reference
- [SLURM & Cluster Deployment](slurm.md) — running VLM jobs at scale
