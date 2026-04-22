# Quick Start

This page walks through the most common MMIRAGE workflows step by step.

## Text-Only: Reformatting a Dataset

Suppose you have a JSONL dataset where each sample looks like:

```json
{
    "conversations": [
        {"role": "user", "content": "Describe the image"},
        {"role": "assistant", "content": "This is a badly formmatted answer"}
    ],
    "modalities": ["<the images>"]
}
```

You want to use an LLM to reformat the assistant answer in Markdown. Create a YAML config:

```yaml
processors:
  - type: llm
    server_args:
      model_path: Qwen/Qwen3-8B
      tp_size: 4
      trust_remote_code: true
    default_sampling_params:
      temperature: 0.1
      top_p: 1.0
      max_new_tokens: 384

loading_params:
  state_dir: /path/to/state
  datasets:
    - path: /path/to/dataset.jsonl
      type: JSONL
      output_dir: /path/to/output/shards
  num_shards: 4
  shard_id: "$SLURM_ARRAY_TASK_ID"
  batch_size: 64

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

  remove_columns: false
  output_schema:
    conversations:
      - role: user
        content: "{{ user_prompt }}"
      - role: assistant
        content: "{{ formatted_answer }}"
    modalities: "{{ modalities }}"

execution_params:
  mode: local
  retry: false
  merge: false
```

Then run:

```bash
mmirage run --config configs/my_config.yaml
```

## Multimodal: Processing Images with a VLM

For vision-language tasks, add image inputs and specify a `chat_template`:

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
  datasets:
    - path: /path/to/image/dataset
      type: loadable
      output_dir: /path/to/output/shards
      image_base_path: /path/to/images
  num_shards: 4
  shard_id: "$SLURM_ARRAY_TASK_ID"
  batch_size: 16

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
        Answer this question about the image:
        {{ question }}

  output_schema:
    question: "{{ question }}"
    answer: "{{ answer }}"

execution_params:
  mode: local
  retry: true
  merge: true
```

## Running on SLURM

Set `execution_params.mode: slurm` and provide SLURM-specific parameters:

```yaml
execution_params:
  mode: slurm
  account: my_account
  job_name: mmirage-job
  nodes: 1
  ntasks_per_node: 1
  gpus: 4
  cpus_per_task: 64
  time_limit: "11:59:59"
  retry: true
  merge: true
  max_retries: 3
```

Then submit:

```bash
mmirage run --config configs/slurm_config.yaml
```

See [CLI Reference](cli.md) for all available subcommands and [Configuration Reference](configuration.md) for a full parameter guide.
