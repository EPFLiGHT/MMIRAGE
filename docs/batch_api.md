# 🗂️ Batch API

This page explains how to run MMIRAGE inference asynchronously using the OpenAI Batch API, which is useful for large-scale processing at lower cost.

---

## Overview

By default MMIRAGE runs in execution_mode in `local`. If you defined execution_mode in `batch`, it will run into the batch processing 

1. **Request serialization:** MMIRAGE serializes inference requests into JSONL chunks.
2. **Batch submission:** Each chunk is uploaded and submitted as an OpenAI batch job.
3. **Execution completion:** The pipeline run exits immediately after submission, saving placeholder values (e.g. `__BATCH_SUBMITTED__:<output_name>:<modality>:<request_number>`) in the output dataset shards.
4. **Asynchronous retrieval:** The user manually polls status and downloads/merges the completed results using separate Python utility modules once the provider completes the batch jobs.

This mode is useful when:

- you do not have access to GPUs locally or on a cluster
- you are processing very large datasets where cost matters
- the pipeline tolerates asynchronous completion (up to 24 h per batch)

---

## When to use each mode

| Criterion | Local (SGLang) | Batch API |
|---|---|---|
| Latency | Low | High (up to 24 h) |
| Cost | GPU compute cost | batch API price |
| GPU requirement | Required | Not required |
| Vision / multimodal | Depends on model | Depends on model |
| Streaming output | ✓ | ✗ |

---

## Configuration

Set `execution_mode` to `batch` and add a `batch` block inside the processor definition in your YAML config:

```yaml
processors:
  - type: llm
    execution_mode: batch
    batch:
      provider: openai
      model: gpt-4o-mini
      max_chunk_bytes: 52428800      # Max bytes per uploaded JSONL file (50 MB)
      max_requests_per_chunk: 50000  # Max requests per batch job
      metadata_output_path: /path/to/batch_metadata.jsonl
      completion_window: 24h
      base_url: https://api.openai.com/v1
      oversized_request_policy: isolate  # isolate | reject
      retry_policy:
        max_attempts: 3
        initial_backoff_seconds: 2.0
        backoff_multiplier: 2.0
```

### Field reference

| Field | Type | Default | Description |
|---|---|---|---|
| `provider` | `str` | — | Provider identifier. Currently `"openai"` is supported. |
| `model` | `str` | `"gpt-4.1-mini"` | Model name for chat completion requests. |
| `batch_endpoint` | `str` | `"/v1/chat/completions"` | Target endpoint used by OpenAI batch jobs. |
| `completion_window` | `str` | `"24h"` | OpenAI batch completion window (only `"24h"` is supported). |
| `max_chunk_bytes` | `int` | `52428800` | Maximum JSONL file size per batch upload (50 MB). |
| `max_requests_per_chunk` | `int` | `null` | Optional hard cap on number of requests in a chunk. |
| `metadata_output_path` | `str` | `""` | Base path for batch job metadata receipt files. Suffixes like `.text.<run_id>.jsonl` and `.multimodal.<run_id>.jsonl` will be appended. |
| `base_url` | `str` | `null` | Optional base URL, useful for API-compatible gateways. |
| `oversized_request_policy` | `str` | `"isolate"` | Policy for requests exceeding `max_chunk_bytes`: `"isolate"` (dedicated chunk) or `"reject"` (fail fast). |
| `retry_policy.max_attempts` | `int` | `3` | Maximum retry attempts for transient submission errors. |
| `retry_policy.initial_backoff_seconds` | `float` | `2.0` | Initial retry delay in seconds. |
| `retry_policy.backoff_multiplier` | `float` | `2.0` | Multiplicative factor for subsequent retry delays. |
| `metadata` | `dict` | `{}` | Key-value pairs sent on batch creation (OpenAI-specific metadata). |
| `credentials.api_key` | `str` | `null` | OpenAI API key (can also be specified via the `OPENAI_API_KEY` env var). |

---

## API key

The OpenAI Batch API requires an API key. You can specify it in your YAML config under `credentials`:

```yaml
    batch_provider:
      provider: openai
      credentials:
        api_key: sk-...
```

Or set it via the environment variable before running:

```bash
export OPENAI_API_KEY=sk-...
mmirage run --config configs/batch_config.yaml
```

MMIRAGE reads the key from either `credentials.api_key` in the config or the `OPENAI_API_KEY` environment variable. Prefer environment variables to avoid committing credentials.

---

## Request chunking

MMIRAGE automatically splits requests into chunks that respect both `max_chunk_bytes` and `max_requests_per_chunk`.

For very large prompts (e.g. with long contexts), you may need to reduce `max_requests_per_chunk` so that individual chunks stay within the size limit. Set `oversized_request_policy: isolate` to submit oversized requests as a dedicated chunk, or `oversized_request_policy: reject` to fail fast on requests exceeding the limit.

---

## Batch Submission & Lifecycle Workflow

Running the batch pipeline is an asynchronous, three-step process:

### Step 1: Submit the Batch Jobs
Execute your MMIRAGE pipeline with a configuration that has `batch_provider.enabled: true`:

```bash
mmirage run --config configs/batch_config.yaml
```

During this run, MMIRAGE maps over your datasets, generates request payloads, writes them to serialized JSONL chunks, and submits them to the OpenAI Batch API.
- The pipeline execution completes immediately after submission.
- The output files in the dataset's `output_dir` shards will contain temporary placeholder variables of the format `__BATCH_SUBMITTED__:<output_name>:<modality>:<request_number>`.
- MMIRAGE generates **metadata receipt files** named `<metadata_output_path>.<modality>.<run_id>.jsonl` (e.g., `batch_metadata.text.abc123.jsonl`). These receipt files store the API batch IDs and map each API request's `custom_id` to its original dataset `source_index`.

### Step 2: Check Batch Job Status
Because batch jobs run asynchronously on the provider's server and can take up to 24 hours to complete, you can monitor their status using the `status_checker` utility module:

```bash
python -m mmirage.core.process.batch.status_checker --config configs/batch_config.yaml
```

By default, the status checker automatically resolves the metadata receipt files from your configuration. You can also specify them manually:

```bash
python -m mmirage.core.process.batch.status_checker \
  --config configs/batch_config.yaml \
  --metadata-path /path/to/batch_metadata.text.abc123.jsonl
```

### Step 3: Retrieve and Merge Results
Once all batch jobs show a status of `completed`, retrieve the generated outputs, map them back to their original row positions, and merge them into a single, ordered JSONL file using the `collector` utility module:

```bash
python -m mmirage.core.process.batch.collector \
  --config configs/batch_config.yaml \
  --output-path /path/to/final_merged_output.jsonl
```

Just like the status checker, the collector automatically locates the metadata receipt files based on the config. To specify the metadata receipts manually, run:

```bash
python -m mmirage.core.process.batch.collector \
  --config configs/batch_config.yaml \
  --metadata-path /path/to/batch_metadata.text.abc123.jsonl \
  --output-path /path/to/final_merged_output.jsonl
```

---

## Provider-Agnostic Architecture & Custom Providers

MMIRAGE's batch processing system is designed to be provider-agnostic. While it comes with built-in support for the OpenAI Batch API, developers can implement custom batch submission providers (such as Anthropic, Mistral, or private gateways) by implementing and registering custom provider configurations and adapters.

### Extension Contracts

To integrate a new provider, you need to implement two classes:

1. **Provider Config Subclass**: Defines the configuration schema. Must inherit from `BatchProviderConfig` ([BatchProviderConfig](../src/mmirage/config/batch_provider.py)).
2. **Submission Adapter Subclass**: Implements request construction, size estimation, chunk submission, status checking, and result retrieval. Must inherit from `BatchSubmissionAdapter` ([BatchSubmissionAdapter](../src/mmirage/core/process/batch/adapter.py#L29)).

#### 1. Custom Provider Config

A custom provider configuration class extends `BatchProviderConfig` with fields specific to that provider:

```python
from dataclasses import dataclass
from mmirage.config.batch_provider import BatchProviderConfig

@dataclass
class AnthropicBatchConfig(BatchProviderConfig):
    provider: str = "anthropic"
    model: str = "claude-haiku-4-5"
```

#### 2. Custom Submission Adapter

A custom adapter implements the core lifecycle logic for the custom provider:

```python
from typing import Any, Dict, Sequence
from mmirage.core.process.batch.adapter import BatchSubmissionAdapter, BatchSubmissionResult
from mmirage.config.batch_provider import BatchProviderConfig

class AnthropicBatchAdapter(BatchSubmissionAdapter):
    # Defines required keys for config.credentials (or environment variable fallbacks)
    required_credentials = ("api_key",)

    def build_request(
        self,
        custom_id: str,
        payload: Dict[str, Any],
        config: BatchProviderConfig,
    ) -> Dict[str, Any]:
        # Formats the internal request payload into the provider's API request format
        ...
        return {
            "custom_id": custom_id,
            "params": {
                "model": config.model,
                "messages": payload["messages"],
            }
        }

    def estimate_request_bytes(self, request: Dict[str, Any]) -> int:
        # Returns the estimated serialized UTF-8 bytes for request size-based chunking
        import json
        return len(json.dumps(request).encode("utf-8"))

    def submit_chunk(
        self,
        chunk_id: str,
        requests: Sequence[Dict[str, Any]],
        config: BatchProviderConfig,
    ) -> Dict[str, Any]:
        # Submits the chunk requests to the provider API and returns the raw response
        ...

    def parse_submission_result(
        self,
        raw_result: Dict[str, Any],
    ) -> BatchSubmissionResult:
        # Wraps the raw submission response in a normalized BatchSubmissionResult
        return BatchSubmissionResult(
            provider_batch_id=raw_result["id"],
            status=raw_result["status"],
            raw_response=raw_result,
        )

    def check_batch_status(
        self,
        provider_batch_id: str,
        config: BatchProviderConfig,
    ) -> BatchSubmissionResult:
        # Queries the provider and returns the latest status
        ...

    def retrieve_results(
        self,
        provider_batch_id: str,
        config: BatchProviderConfig,
    ) -> Sequence[Dict[str, Any]]:
        # Downloads/retrieves completed outputs and normalizes each row.
        # Ensure text generations are mapped to the "generated_text" key so
        # the collector can reconstruct the original dataset rows neutrally.
        ...
```

### Registry Integration

Once you have defined your config and adapter classes, register them with the MMIRAGE batch system at runtime (typically inside your application's bootstrap or initialization code):

```python
from mmirage.core.process.batch.provider_resolution import BatchProviderConfigRegistry
from mmirage.core.process.batch.registry import BatchAdapterRegistry

# Register the provider configuration class
BatchProviderConfigRegistry.register("anthropic", AnthropicBatchConfig)

# Register the provider submission adapter
BatchAdapterRegistry.register("anthropic", AnthropicBatchAdapter)
```

### Config Usage

After registering your custom provider, you can reference it in your MMIRAGE pipeline YAML configuration:

```yaml
processors:
  - type: llm
    batch_provider:
      provider: anthropic
      enabled: true
      model: claude-haiku-4-5
      metadata_output_path: /scratch/anthropic_meta.jsonl
      credentials:
        api_key: "your-anthropic-key"  # Or leave blank and set ANTHROPIC_API_KEY env var
```

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
      max_chunk_bytes: 52428800
      max_requests_per_chunk: 50000
      metadata_output_path: /scratch/batch_meta.jsonl
      completion_window: 24h
      base_url: https://api.openai.com/v1
      oversized_request_policy: isolate
      retry_policy:
        max_attempts: 3
        initial_backoff_seconds: 2.0
        backoff_multiplier: 2.0

loading_params:
  state_dir: /scratch/state
  datasets:
    - path: /data/my_dataset.jsonl
      type: JSONL
      output_dir: /scratch/output/shards
  num_shards: 1
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
- [CLI Reference](cli.md) — CLI command reference for local and SLURM pipeline execution
