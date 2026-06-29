# ⚙️ Configuration Reference

This page is the complete reference for every parameter in the MMIRAGE YAML configuration file.

A pipeline config is split into four top-level sections: `processors`, `loading_params`,
`processing_params`, and `execution_params`.

To route inference through the OpenAI Batch API, configure `execution_mode` under your `llm` processor (see `processors[*].batch_provider` below).
If you are new to MMIRAGE, read [Concepts](concepts.md) first to understand the terminology,
then follow [Quickstart](quickstart.md) for a minimal working example.

---

## `processors`



A list of processor definitions. Currently the only supported type is `llm`.

The processor `llm` has to two `execution_mode` : 
- `local` (by default): run on your hardware
- `batch` : run thanks to a API LLM provider.
 
### `processors[*].local` — Local execution mode

```yaml
processors:
  - type: llm
    execution_mode: local
    local:
      server_args:
        model_path: Qwen/Qwen3-8B
        tp_size: 4
        trust_remote_code: true
        disable_custom_all_reduce: false
      chat_template: ""           # Set to e.g. "qwen2-vl" for VLMs
      default_sampling_params:
        temperature: 0.1
        top_p: 0.9
        max_new_tokens: 1024
```

#### `processors[*].server_args`

| Field | Type | Default | Description |
|---|---|---|---|
| `model_path` | `str` | `"none"` | HuggingFace model ID or local path |
| `tp_size` | `int` | auto from `SLURM_GPUS_ON_NODE` | Tensor parallelism size |
| `trust_remote_code` | `bool` | `true` | Allow custom model code from HuggingFace |
| `disable_custom_all_reduce` | `bool` | `false` | Disable custom all-reduce kernel |
| `extra_engine_args` | `dict` | `{}` | Additional keyword arguments forwarded verbatim to `sgl.Engine` |

Use `extra_engine_args` to pass SGLang engine options not listed above:

```yaml
server_args:
  model_path: Qwen/Qwen3-8B
  tp_size: 4
  extra_engine_args:
    max_running_requests: 1000
    chunked_prefill_size: 32768
    mem_fraction_static: 0.88
```

#### `processors[*].default_sampling_params`

Any key-value pairs accepted by the SGLang sampling API, e.g.:

| Field | Description |
|---|---|
| `temperature` | Sampling temperature |
| `top_p` | Top-p nucleus sampling |
| `max_new_tokens` | Maximum tokens to generate |

Additional model-specific options can be passed under `custom_params`:

```yaml
default_sampling_params:
  temperature: 0.1
  top_p: 0.9
  max_new_tokens: 1024
  custom_params:
    chat_template_kwargs:
      enable_thinking: false   # Qwen3 thinking-mode control
```

#### `processors[*].chat_template`

Optional. Set to a named template (e.g. `qwen2-vl`, `llava`, `internvl`, `phi3_v`) for vision-language models. Defaults to the tokenizer's built-in template.

### `processors[*].batch` — Batch execution mode
Optional. When set, MMIRAGE routes requests through the OpenAI Batch API instead of running a local SGLang server. This is useful for large-scale processing without a local GPU.

```yaml
processors:
  - type: llm
    execution_mode: batch
    batch:
      provider: openai
      model: gpt-4o-mini
      max_chunk_bytes: 52428800    # 50 MB per batch file
      metadata_output_path: /path/to/batch_metadata.jsonl
```

**`batch` fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `provider` | `str` | — | Provider identifier. Currently `"openai"` is supported |
| `model` | `str` | `gpt-4.1-mini` | Model name for chat completion requests |
| `max_chunk_bytes` | `int` | `52428800` | Max serialized bytes per batch file (50 MB) |
| `max_requests_per_chunk` | `int` | `null` | Optional hard cap on requests per chunk |
| `metadata_output_path` | `str` | `""` | Base path for submission receipt files |
| `completion_window` | `str` | `"24h"` | OpenAI batch completion window |
| `base_url` | `str` | `null` | Optional base URL for API-compatible gateways |
| `oversized_request_policy` | `str` | `"isolate"` | `"isolate"` or `"reject"` for requests exceeding `max_chunk_bytes` |
| `retry_policy.max_attempts` | `int` | `3` | Max retry attempts for transient submission errors |
| `retry_policy.initial_backoff_seconds` | `float` | `2.0` | Initial retry delay |
| `retry_policy.backoff_multiplier` | `float` | `2.0` | Multiplicative factor for subsequent retry delays |

---

## `loading_params`

Controls how datasets are loaded and distributed across shards.

```yaml
loading_params:
  state_dir: ~/.cache/MMIRAGE/state_dir
  datasets:
    - path: /path/to/data.jsonl
      type: JSONL
      output_dir: /path/to/output/shards
      image_base_path: /path/to/images   # optional, for vision tasks
  num_shards: 4
  shard_id: "$SLURM_ARRAY_TASK_ID"
  batch_size: 64
```

| Field | Type | Default | Description |
|---|---|---|---|
| `state_dir` | `str` | `~/.cache/MMIRAGE/state_dir` | Shared directory for shard state, retry markers, and status files |
| `datasets` | `list` | `[]` | List of dataset configurations (see below) |
| `num_shards` | `int` or env var | `1` | Total number of shards to split datasets into |
| `shard_id` | `int` or env var | `0` | Index of this shard (0-based). In SLURM use `"$SLURM_ARRAY_TASK_ID"` |
| `batch_size` | `int` | `1` | Batch size for processing samples |

### `loading_params.datasets[*]`

| Field | Type | Required | Description |
|---|---|---|---|
| `path` | `str` | ✓ | Path to dataset file or directory |
| `type` | `str` | ✓ | Loader type: `JSONL` or `loadable` (HuggingFace `load_from_disk`) |
| `output_dir` | `str` | ✓ | Directory where processed shards are written |
| `image_base_path` | `str` | — | Base directory for resolving relative image paths |

---

## `processing_params`

Defines variable extraction, LLM-driven generation, and the final output structure.

```yaml
processing_params:
  inputs:
    - name: my_var
      key: field.nested[0].value    # JMESPath expression
      type: text                    # "text" (default) or "image"

  outputs:
    - name: my_output
      type: llm
      output_type: plain            # "plain" or "JSON"
      prompt: |
        Do something with {{ my_var }}
      output_schema:                # Only for output_type: JSON
        - field_a
        - field_b

  remove_columns: false
  output_schema:
    result: "{{ my_output }}"
```

### `processing_params.inputs[*]`

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | — | Variable name used in Jinja2 templates |
| `key` | `str` | — | JMESPath expression to extract value from a sample |
| `type` | `str` | `text` | `"text"` or `"image"`. Image variables are resolved to PIL Images / absolute paths |

### `processing_params.outputs[*]`

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | — | Variable name made available in `output_schema` templates |
| `type` | `str` | — | Processor type — must match a registered processor (`llm`) |
| `output_type` | `str` | `plain` | `"plain"` (raw text) or `"JSON"` (structured object) |
| `prompt` | `str` | — | Jinja2 template for the LLM prompt |
| `output_schema` | `list[str]` | `[]` | Required field names when `output_type: JSON` |

### `processing_params.output_schema`

A dictionary describing the structure of each output sample. Values are Jinja2 templates that reference input or output variable names. Nested dicts and lists are supported.

### `processing_params.remove_columns`

If `true`, all original columns are removed from the dataset before writing; only columns defined in `output_schema` are kept. Defaults to `false`.

---

## `execution_params`

Controls where and how the pipeline runs.

```yaml
execution_params:
  mode: local           # "local" or "slurm"
  retry: false
  merge: false
  max_retries: 3
  poll_interval_seconds: 30
  settle_time_seconds: 60

  # SLURM-specific (required when mode: slurm)
  account: my_account
  job_name: mmirage-sharded
  reservation: ""
  nodes: 1
  ntasks_per_node: 1
  gpus: 4
  cpus_per_task: 288
  time_limit: "11:59:59"

  # Paths
  project_root: /path/to/project   # Supports ${ENV_VAR} expansion
  report_dir: ~/reports
  hf_home: ~/hf
  edf_env: ""
```

### Core fields

| Field | Type | Default | Description |
|---|---|---|---|
| `mode` | `str` | `local` | `"local"` (run in-process) or `"slurm"` (submit sbatch array job) |
| `retry` | `bool` | `false` | Auto-retry failed shards until success or `max_retries` is reached |
| `merge` | `bool` | `false` | Merge shard outputs after a successful run |
| `max_retries` | `int` | `3` | Maximum retries per shard |
| `poll_interval_seconds` | `int` | `30` | Seconds between SLURM job status polls |
| `settle_time_seconds` | `int` | `60` | Seconds to wait after a SLURM job finishes before checking shard state |

### SLURM-specific fields

| Field | Type | Default | Description |
|---|---|---|---|
| `account` | `str` | — | HPC account/partition (**required** for SLURM mode) |
| `job_name` | `str` | `mmirage-sharded` | SLURM job name |
| `reservation` | `str` | — | Optional SLURM reservation |
| `nodes` | `int` | `1` | Number of nodes |
| `ntasks_per_node` | `int` | `1` | Tasks per node |
| `gpus` | `int` | `4` | GPUs per node |
| `cpus_per_task` | `int` | `288` | CPUs per task |
| `time_limit` | `str` | `11:59:59` | Wall-clock time limit (`HH:MM:SS`) |

### Path fields

| Field | Type | Default | Description |
|---|---|---|---|
| `project_root` | `str` | — | Base project directory. Supports `${VAR}` expansion |
| `report_dir` | `str` | `~/reports` | Directory for SLURM stdout/stderr logs |
| `hf_home` | `str` | `~/hf` | HuggingFace cache directory |
| `edf_env` | `str` | — | Optional EDF environment file path |

---

## Merge output behaviour

| Trigger | Merged output location |
|---|---|
| `run` with `merge: true` | `<dataset.output_dir>/merged/` per dataset |
| `merge` without `--output-root` | `<dataset.output_dir>/merged/` per dataset |
| `merge --output-root /path` | `/path/<dataset_name>/` per dataset |
| `merge-dir --input-dir /path --output-dir /out` | `/out/` (single dataset) |

If `shard_*` folders are present **directly** inside `--input-dir`, MMIRAGE merges that dataset and ignores nested subdirectories (e.g. `_pipeline_state`).

---

## See also

- [Concepts](concepts.md) — vocabulary for all parameters on this page
- [Quickstart](quickstart.md) — minimal working config examples
- [Multimodal Processing](multimodal.md) — image inputs and `chat_template`
- [SLURM & Cluster Deployment](slurm.md) — `execution_params` for SLURM mode
- [Batch API](batch_api.md) — `batch_provider` configuration in depth
- [CLI Reference](cli.md) — how to run a configured pipeline
