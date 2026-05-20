# 🗂️ Batch API

This page explains how to run MMIRAGE inference asynchronously using the
OpenAI Batch API, which is useful for large-scale processing at lower cost.

---

## Overview

By default, MMIRAGE runs inference locally via an SGLang engine.
When a `batch_provider` is configured, the `llm` processor instead delegates
requests to the OpenAI Batch API:

1. Requests are serialised into JSONL chunks.
2. Each chunk is uploaded and submitted as an OpenAI batch job.
3. MMIRAGE polls the API until all batches complete.
4. Responses are collected, validated, and returned to the mapper.

This mode is useful when:

- you do not have access to GPUs locally or on a cluster
- you are processing very large datasets where cost matters
- the pipeline tolerates asynchronous completion (up to 24 h per batch)

---

## When to use each mode

| Criterion | Local (SGLang) | Batch API |
|---|---|---|
| Latency | Low (minutes per shard) | High (up to 24 h) |
| Cost | GPU compute cost | ~50 % lower per token |
| GPU requirement | Required | Not required |
| Vision / multimodal | ✓ | Depends on model |
| Streaming output | ✓ | ✗ |

---

## Configuration

Add a `batch_provider` section to your YAML config:

```yaml
batch_provider:
  provider: openai
  enabled: true
  model: gpt-4o-mini
  max_chunk_bytes: 95000000      # Max bytes per uploaded JSONL file (~95 MB)
  max_requests_per_chunk: 50000  # Max requests per batch job
  metadata_output_path: /path/to/batch_metadata.json
  completion_window: 24h
  base_url: https://api.openai.com/v1
  oversized_request_policy: warn  # warn | skip | error
  retry_policy:
    max_retries: 3
    initial_delay: 5.0
    backoff_factor: 2.0
    max_delay: 300.0
```

### Field reference

| Field | Type | Description |
|---|---|---|
| `provider` | `str` | Always `openai` |
| `enabled` | `bool` | Enable batch mode (default `false`) |
| `model` | `str` | OpenAI model ID (e.g. `gpt-4o-mini`, `gpt-4o`) |
| `max_chunk_bytes` | `int` | Maximum JSONL file size per batch upload |
| `max_requests_per_chunk` | `int` | Maximum requests per batch job |
| `metadata_output_path` | `str` | Path to write batch job metadata JSON |
| `completion_window` | `str` | OpenAI batch window (`24h`) |
| `base_url` | `str` | OpenAI API base URL |
| `oversized_request_policy` | `str` | Behaviour for requests exceeding size limits: `warn`, `skip`, or `error` |
| `retry_policy.max_retries` | `int` | Maximum polling retries on transient errors |
| `retry_policy.initial_delay` | `float` | Initial polling delay in seconds |
| `retry_policy.backoff_factor` | `float` | Exponential backoff multiplier |
| `retry_policy.max_delay` | `float` | Maximum delay between polls in seconds |

---

## API key

The OpenAI Batch API requires an API key.
Set it via environment variable before running:

```bash
export OPENAI_API_KEY=sk-...
mmirage run --config configs/batch_config.yaml
```

MMIRAGE never reads the key from the config file — always use environment variables
to avoid accidentally committing credentials.

---

## Request chunking

MMIRAGE automatically splits requests into chunks that respect both
`max_chunk_bytes` and `max_requests_per_chunk`.

For very large prompts (e.g. with long contexts), you may need to reduce
`max_requests_per_chunk` so that individual chunks stay within the size limit.
Set `oversized_request_policy: skip` to drop single requests that exceed the
limit rather than failing the entire batch.

---

## Monitoring batch jobs

MMIRAGE prints progress as batches are submitted and polled.
The `metadata_output_path` file records all submitted batch job IDs and their
final status, which can be useful for debugging or manual inspection.

---

## Complete example config

```yaml
processors:
  - type: llm
    server_args:
      model_path: none        # Ignored in batch mode
    default_sampling_params:
      temperature: 0.0
      max_new_tokens: 512

batch_provider:
  provider: openai
  enabled: true
  model: gpt-4o-mini
  max_chunk_bytes: 95000000
  max_requests_per_chunk: 50000
  metadata_output_path: /scratch/batch_meta.json
  completion_window: 24h
  base_url: https://api.openai.com/v1
  oversized_request_policy: warn
  retry_policy:
    max_retries: 3
    initial_delay: 5.0
    backoff_factor: 2.0
    max_delay: 300.0

loading_params:
  state_dir: /scratch/state
  datasets:
    - path: /data/my_dataset.jsonl
      type: JSONL
      output_dir: /scratch/output/shards
  num_shards: 4
  shard_id: 0
  batch_size: 512

processing_params:
  inputs:
    - name: question
      key: question

  outputs:
    - name: answer
      type: llm
      output_type: plain
      prompt: |
        Answer the following question concisely:
        {{ question }}

  output_schema:
    question: "{{ question }}"
    answer: "{{ answer }}"

execution_params:
  mode: local
  retry: false
  merge: false
```

---

## See also

- [Concepts](concepts.md) — processor types and execution modes
- [Configuration Reference](configuration.md) — full `batch_provider` parameter reference
- [Pipeline](pipeline.md) — where batch inference fits in the data flow
- [CLI Reference](cli.md) — `submit`, `check`, and `retry` for batch workflows
