# Image Generation

This page explains how to generate one image per dataset sample with the
`image_gen` processor. For the general pipeline structure, input extraction,
and output rendering, read [Concepts](concepts.md) and [Pipeline](pipeline.md).

---

## Overview

The image-generation processor renders a Jinja2 prompt for every sample, sends
it to an HTTP image-generation backend, and returns either a saved image path or
a PIL image. It supports two backend modes:

| Backend | Server lifecycle |
|---|---|
| `external` | You start and stop the HTTP server and provide its URL |
| `sglang` | MMIRAGE starts one shared SGLang Diffusion server for the run, waits for it to become ready, and stops it afterward |

Both modes send requests to the OpenAI-compatible
`POST /v1/images/generations` endpoint and expect a base64 image in
`data[0].b64_json`.

---

## Installation

The base MMIRAGE installation contains the HTTP client and image-processing
dependencies needed to use an external server.

To let MMIRAGE launch SGLang Diffusion itself, install the image-generation
extra:

```bash
pip install -e ".[image_gen]"
```

This installs `sglang[diffusion]==0.5.10`. The `sglang` executable must be
available in the active environment when the pipeline starts.

---

## External server example

The following local pipeline reads prompts from a JSONL file and stores the
generated images on disk:

```yaml
processors:
  - type: image_gen
    backend: external
    external:
      base_url: http://127.0.0.1:30010/v1
      timeout_seconds: 900
      max_concurrent_requests: 4
    default_sampling_params:
      num_inference_steps: 30
      guidance_scale: 4.0
    parallel_inference: true
    parallel_chunk_size: 4
    output_dir: /path/to/generated/images
    file_format: png

loading_params:
  state_dir: /path/to/state
  datasets:
    - path: /path/to/prompts.jsonl
      type: JSONL
      output_dir: /path/to/output/shards
  num_shards: 1
  shard_id: 0
  batch_size: 8

processing_params:
  inputs:
    - name: text
      key: text

  outputs:
    - name: generated_image
      type: image_gen
      output_mode: path
      filename_template: "img_{{ __shard_id }}_{{ __sample_index }}_{{ __source_hash }}"
      width: 1024
      height: 1024
      seed: 42
      prompt: |
        A photorealistic image of: {{ text }}

  output_schema:
    caption: "{{ text }}"
    image: "{{ generated_image }}"

execution_params:
  mode: local
  retry: false
  merge: false
```

Run it with:

```bash
mmirage run --config /path/to/config.yaml
```

A runnable configuration with the same structure is available at
`configs/config_image_gen_external.yaml`.

---

## MMIRAGE-managed SGLang server

Use `backend: sglang` when MMIRAGE should manage the server process:

```yaml
processors:
  - type: image_gen
    backend: sglang
    sglang:
      model_path: Qwen/Qwen-Image
      num_gpus: 1
      startup_timeout_seconds: 900
      timeout_seconds: 900
      max_concurrent_requests: 4
      api_key: EMPTY
    default_sampling_params:
      num_inference_steps: 30
      guidance_scale: 4.0
    parallel_inference: true
    parallel_chunk_size: 4
    output_dir: /path/to/generated/images
    file_format: png
```

MMIRAGE selects an available localhost port, starting at `30010` unless `port`
is configured. It launches `sglang serve`, waits for a health or models endpoint,
shares the resolved URL with shard workers, and stops the process when the run
ends. In SLURM mode, the shard workers in the job share that server. A config may
contain at most one `image_gen` processor with `backend: sglang`.

See `configs/config_image_gen_sglang.yaml` for a complete SLURM example.

### SGLang backend fields

| Field | Type | Default | Description |
|---|---|---|---|
| `model_path` | `str` | required | Model path passed to `sglang serve --model-path` |
| `port` | `int` or `null` | `null` | Preferred localhost port; automatic selection starts at `30010` when omitted |
| `num_gpus` | `int` | `1` | Value passed to `sglang serve --num-gpus` |
| `dtype` | `str` or `null` | `null` | Optional value passed to `sglang serve --dtype` |
| `startup_timeout_seconds` | `int` | `900` | Maximum time to wait for server readiness |
| `extra_server_args` | `list[str]` | `[]` | Additional command-line arguments appended to `sglang serve` |
| `api_key` | `str` or `null` | `null` | Bearer token used by readiness checks and generation requests |
| `timeout_seconds` | `int` | `900` | Timeout for each generation request |
| `request_model` | `str` or `null` | `null` | Optional `model` field included in generation requests |
| `max_concurrent_requests` | `int` | `1` | Maximum simultaneous HTTP generation requests |

---

## Processor fields

These fields apply to both backend modes:

| Field | Type | Default | Description |
|---|---|---|---|
| `default_sampling_params` | `dict` | `{}` | Default parameters included in every generation request |
| `parallel_inference` | `bool` | `true` | Try each multi-sample chunk through the backend's batch interface |
| `parallel_chunk_size` | `int` or `null` | `4` | Samples per chunk; `null` uses the full mapper batch |
| `output_dir` | `str` | `~/.cache/MMIRAGE/generated_images` | Directory used by outputs with `output_mode: path` |
| `file_format` | `str` | `png` | Extension and PIL save format for generated files |

For `backend: external`, configure these client fields:

| Field | Type | Default | Description |
|---|---|---|---|
| `base_url` | `str` | required | Server root URL or URL ending in `/v1` |
| `api_key` | `str` or `null` | `null` | Optional bearer token |
| `timeout_seconds` | `int` | `900` | Timeout for each generation request |
| `request_model` | `str` or `null` | `null` | Optional `model` field included in requests |
| `max_concurrent_requests` | `int` | `1` | Maximum simultaneous HTTP generation requests |

The client checks the server when the processor is initialized. It tries the
server's `/models`, `/v1/models`, and `/health` endpoints.

### Sampling parameters

`default_sampling_params` are shared by all `image_gen` outputs using that
processor. The processor explicitly supports `num_inference_steps`,
`guidance_scale`, image size, `output_quality`/`output-quality`, and
`output_compression`/`output-compression`. Other non-reserved keys are forwarded
to the server unchanged.

The backend currently requires one image per request (`n: 1`) and decodes only
base64 responses, so `response_format` must remain `b64_json`. A `size` sampling
parameter can be supplied directly; alternatively, setting both `width` and
`height` produces a `<width>x<height>` size string.

---

## Image output fields

Each `processing_params.outputs` entry with `type: image_gen` accepts:

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Variable name used in `output_schema` |
| `prompt` | `str` | `""` | Jinja2 positive-prompt template |
| `negative_prompt` | `str` | `""` | Optional Jinja2 negative-prompt template |
| `output_mode` | `path` or `pil` | `path` | Return an absolute saved path or a PIL image |
| `filename_template` | `str` | `generated_{{ __shard_id }}_{{ __sample_index }}_{{ __source_hash }}` | Filename stem template used in `path` mode |
| `width` | `int` or `null` | `null` | Per-output width override |
| `height` | `int` or `null` | `null` | Per-output height override |
| `num_inference_steps` | `int` or `null` | `null` | Per-output sampling-step override |
| `guidance_scale` | `float` or `null` | `null` | Per-output guidance-scale override |
| `seed` | `int` or `null` | `null` | Base seed for deterministic, shard-aware per-sample seeds |

Per-output width, height, inference steps, and guidance scale override values
with the same keys in `default_sampling_params`.

### Filename templates

The filename template can use all extracted input variables plus these internal
variables:

| Variable | Value |
|---|---|
| `__sample_index` | Zero-based position within the current shard's output |
| `__output_name` | Name of the current output variable |
| `__shard_id` | Current shard ID |
| `__source_hash` | First eight hexadecimal characters of a SHA-256 hash of the input values |

Rendered stems are restricted to letters, digits, `.`, `_`, and `-`; other
character runs become `_`. Include `__shard_id` when multiple shards write to
the same image directory. If a target path already exists, MMIRAGE preserves it
and adds a hostname, process ID, and run token to the new filename.

---

## Stored image representation

With `output_mode: path`, MMIRAGE saves each image atomically under the
processor's `output_dir` and initially places its absolute path in the generated
variable.

By default, `processing_params.cast_images` is `true`. When an `output_schema`
field is a direct reference such as `image: "{{ generated_image }}"`, MMIRAGE
casts that path column to the Hugging Face `Image` feature before saving the
dataset shard. Hugging Face then embeds the image bytes in the Arrow shard. Set
`cast_images: false` to retain plain path strings:

```yaml
processing_params:
  cast_images: false
  outputs:
    - name: generated_image
      type: image_gen
      output_mode: path
      prompt: "{{ text }}"
  output_schema:
    image_path: "{{ generated_image }}"
```

With `output_mode: pil`, the generated variable contains the PIL image directly
and no separate file is written by the processor.

---

## Parallel requests and failures

When `parallel_inference` is enabled for a batch containing more than one
sample, MMIRAGE splits it into chunks of `parallel_chunk_size`. Within a chunk,
the HTTP backend sends up to `max_concurrent_requests` requests concurrently.

If a chunk-level call fails, MMIRAGE retries only that chunk one sample at a
time. An individual sample that still fails receives `None` as its generated
value, while processing continues for the other samples. Setting
`parallel_inference: false` uses the per-sample path from the start.

---

## See also

- [Pipeline](pipeline.md) — how processor outputs flow into the saved dataset
- [Configuration Reference](configuration.md) — shared loading, processing, and execution parameters
- [SLURM & Cluster Deployment](slurm.md) — running sharded jobs on a cluster
- [Benchmarking](benchmarking.md) — shard runtime and GPU metrics
