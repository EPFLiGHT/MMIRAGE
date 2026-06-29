# 💡 Concepts

This page defines the core vocabulary used throughout the MMIRAGE documentation.
Read it before diving into the pipeline or configuration details.

---

## Dataset and samples

A **dataset** is any collection of data samples that MMIRAGE reads as input.
MMIRAGE supports two dataset formats:

- **JSONL** — a plain text file where each line is a JSON object representing one sample.
- **Loadable** — a HuggingFace-compatible dataset directory saved with `save_to_disk`.

Each **sample** is one record in the dataset, typically a JSON object with fields like
`question`, `answer`, `conversations`, `image_path`, or whatever your data contains.

---

## Shards

MMIRAGE splits a dataset into **shards** — non-overlapping partitions that are
processed independently and in parallel.

The total number of shards is set by `loading_params.num_shards`.
Each shard is identified by a `shard_id` (0-indexed).
On SLURM, `shard_id` is typically set to `$SLURM_ARRAY_TASK_ID` so that each
array task processes exactly one shard.

The processed output of each shard is written to a separate `shard_<id>/`
subdirectory under `output_dir`.

---

## State directory

The **state directory** is a folder that tracks the status of each shard.
After a shard finishes, MMIRAGE writes a `status.json` file there recording
whether the shard succeeded or failed, and how many attempts it took.

The state directory enables:

- **Resume** — skip already-succeeded shards when re-running.
- **Retry** — automatically resubmit failed shards.

Set the state directory with `loading_params.state_dir`.

---

## Input variables

An **input variable** is a named value extracted from each data sample.
Variables are defined in `processing_params.inputs` and extracted using
**JMESPath** expressions.

For example:

```yaml
inputs:
  - name: question
    key: conversations[0].content
  - name: answer
    key: conversations[1].content
  - name: image
    key: image_path
    type: image
```

The `key` field is a JMESPath expression applied to each sample.
Image variables (`type: image`) are automatically resolved to PIL Images
or absolute file paths, using `image_base_path` if set.

---

## Output variables

An **output variable** is the result of running the processor on a prompt.
Output variables are defined in `processing_params.outputs`.

Each output variable specifies:

- a **name** — how it is referenced in the output schema
- a **type** — always `llm` for the current processor
- an **output_type** — `plain` for raw text, `JSON` for parsed JSON
- a **prompt** — a Jinja2 template that constructs the message sent to the model

```yaml
outputs:
  - name: formatted_answer
    type: llm
    output_type: plain
    prompt: |
      Reformat this answer in Markdown:
      {{ answer }}
```

Inside the prompt, you can reference any input variable by name using `{{ variable }}`.

---

## JMESPath

**JMESPath** is a query language for extracting values from JSON.
MMIRAGE uses it in `inputs[*].key` to pull fields out of each sample.

Common patterns:

| Expression | Meaning |
|---|---|
| `field` | Top-level field named `field` |
| `nested.field` | Nested field access |
| `list[0].content` | First element of a list, then a subfield |
| `list[-1].content` | Last element of a list |

MMIRAGE compiles and caches JMESPath expressions at startup to avoid
recompilation on every sample.

---

## Jinja2 templates

**Jinja2** is a templating language used in two places in MMIRAGE:

1. **Prompts** (`outputs[*].prompt`) — to construct the message sent to the model.
2. **Output schema** (`processing_params.output_schema`) — to render the final saved sample.

In both cases, extracted input variables and generated output variables are
available as template variables using `{{ variable_name }}`.

Simple `{{ var }}` references in the output schema bypass Jinja2 rendering
to preserve non-string types (e.g. PIL Images, lists, dicts).

---

## Output schema

The **output schema** defines the structure of each saved sample.
It is a YAML object under `processing_params.output_schema` that uses
Jinja2 `{{ variable }}` references.

For example:

```yaml
output_schema:
  conversations:
    - role: user
      content: "{{ question }}"
    - role: assistant
      content: "{{ formatted_answer }}"
  image_path: "{{ image }}"
```

The output schema controls exactly what fields end up in the processed dataset.
Fields not listed in the schema are dropped unless `remove_columns: false` is set,
in which case all original fields are kept alongside the new outputs.

---

## Processor

A **processor** is the inference engine that generates outputs.
Currently, MMIRAGE supports one processor type: `llm`.

The `llm` processor runs a language model via one of two backends:

- **Local inference** — starts an SGLang engine on the current machine (or SLURM node).
- **Batch API** — sends requests asynchronously to the OpenAI Batch API
  (configured via `batch_provider`; see [Batch API](batch_api.md)).

---

## Execution modes

- **`local`** — runs a single shard in the current Python environment (defaults to shard 0).
  Use `mmirage run --shard-id N` to select a shard; use `--force-retry` (or `execution_params.retry: true`) to iterate over all shards locally.
- **`slurm`** — MMIRAGE generates and submits an `sbatch` array job.
  Each array task processes one shard on a dedicated node.

See [SLURM & Cluster Deployment](slurm.md) for details on the SLURM workflow.

---

## Retry and merge

After processing, MMIRAGE can automatically:

- **Retry** failed shards up to `max_retries` times (set `retry: true`).
- **Merge** all shard outputs into a single dataset directory at
  `<output_dir>/merged/` (set `merge: true`).

Both options are controlled under `execution_params`.

---

## See also

- [Pipeline](pipeline.md) — step-by-step walkthrough of the data flow
- [Configuration Reference](configuration.md) — complete parameter reference
- [Quickstart](quickstart.md) — run a first pipeline with these concepts in practice
