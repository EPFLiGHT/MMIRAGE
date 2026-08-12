# 🏗️ Architecture

This page covers MMIRAGE's internal module layout and the key design decisions behind each subsystem.
It is aimed at contributors and developers who want to understand or modify the codebase.

If you are looking for a user-facing explanation of what happens when you run `mmirage run`,
read [Pipeline](pipeline.md) instead.

---

## High-Level Overview

Each shard follows the same three-stage pipeline:

```{image} _static/pipeline_diagram.png
:alt: MMIRAGE pipeline — Loading Data → Processor → Write Data
:align: center
:width: 90%
```

At the orchestration level, the CLI manages shard dispatch, retry logic, and optional SLURM submission:

```
               ┌─────────────────────────────────────────────────────┐
               │                    mmirage CLI                       │
               │  run / submit / check / retry / merge / stats        │
               └───────────────────┬─────────────────────────────────┘
                                   │
              ┌────────────────────▼───────────────────────┐
              │              launch_pipeline                │
              │  (local loop or SLURM array submission)     │
              └──────────────┬─────────────────────────────┘
                             │  spawns
              ┌──────────────▼─────────────────────────────┐
              │           shard_process.py                  │
              │  (one process per shard)                    │
              └──┬──────────┬──────────────┬───────────────┘
                 │          │              │
        ┌────────▼──┐  ┌────▼────┐  ┌────▼──────┐
        │  Loader   │  │ Mapper  │  │ Renderer  │
        │(Dataset)  │  │(Compute)│  │(Jinja2)   │
        └────────┬──┘  └────┬────┘  └────┬──────┘
                 │          │            │
                 └──────────┴────────────┘
                            │
              ┌─────────────▼─────────────┐
              │     shard_utils.py        │
              │  atomic save + state mgmt │
              └───────────────────────────┘
```

---

## Package Layout

```
src/mmirage/
├── __init__.py              Public API surface (MMirageConfig, load_mmirage_config …)
├── cli.py                   CLI entry point and subcommand handlers
├── shard_process.py         Single-shard processing script
├── shard_utils.py           Shard state, atomic saves, GPU polling, benchmarking
├── merge_shards.py          Post-processing: merge shard_* dirs into one dataset
│
├── config/                  Configuration layer (pure dataclasses, no heavy deps)
│   ├── config.py            MMirageConfig, ExecutionParams, ProcessingParams
│   ├── loading.py           LoadingParams, env-var resolution
│   ├── batch_provider.py    Provider-neutral BatchProviderConfig
│   ├── openai_batch.py      OpenAIBatchConfig (extends BatchProviderConfig)
│   ├── anthropic_batch.py   AnthropicBatchConfig (extends BatchProviderConfig)
│   └── utils.py             YAML loader, env-var expansion, dacite wiring
│
├── cli_utils/               CLI helpers
│   ├── runtime.py           Path expansion, file logging, setup_runtime
│   ├── slurm.py             sbatch script generation, job submission, polling
│   └── status.py            Shard status reads, retry budget, check_failed_shards
│
└── core/
    ├── loader/              Dataset loading
    │   ├── base.py          BaseDataLoader, DataLoaderRegistry
    │   ├── jsonl.py         JSONL loader (type: "JSONL")
    │   ├── local_hf.py      HuggingFace load_from_disk (type: "loadable")
    │   └── utils.py         load_datasets_from_configs helper
    │
    ├── process/             Data transformation
    │   ├── variables.py     InputVar, OutputVar, VariableEnvironment, JMESPath cache
    │   ├── base.py          BaseProcessor, ProcessorRegistry, TokenCounts
    │   ├── mapper.py        MMIRAGEMapper — orchestrates variables through processors
    │   ├── processors/
    │   │   ├── llm/
    │   │   │   ├── config.py         SGLangLLMConfig, SGLangServerArgs, LLMOutputVar
    │   │   │   └── llm_processor.py  LLMProcessor — SGLang engine wrapper
    │   │   └── batch_api/
    │   │       ├── config.py              BatchApiProcessorConfig
    │   │       └── batch_api_processor.py BatchApiProcessor — provider batch submission
    │   └── batch/           Async/batch inference subsystem
    │       ├── orchestrator.py       End-to-end batch pipeline
    │       ├── adapter.py            Provider-neutral batch adapter interface
    │       ├── openai_adapter.py     OpenAI Batch API adapter
    │       ├── anthropic_adapter.py  Anthropic Messages Batches adapter
    │       ├── chunking.py           Request chunking (byte/count limits)
    │       ├── collector.py          Response collection and result joining
    │       ├── status_checker.py     Batch job polling
    │       └── registry.py           Adapter registry
    │
    └── writer/
        └── renderer.py      TemplateRenderer — Jinja2 output_schema rendering
```

---

## Data Flow

### Single Shard (local mode)

1. **Config loading** — `load_mmirage_config` reads the YAML, expands `${ENV_VAR}` references, and constructs a typed `MMirageConfig` via `dacite`.
2. **Dataset loading** — `load_datasets_from_configs` calls the appropriate `DataLoader` (JSONL or `loadable`), returning a HuggingFace `Dataset`.
3. **Sharding** — The dataset is split into `num_shards` slices; this shard processes slice `shard_id`.
4. **Mapping** — `MMIRAGEMapper.rewrite_batch` iterates over batches:
   - Extracts `InputVar` values from each sample using cached JMESPath expressions.
   - Resolves image inputs to PIL Images or absolute paths.
   - Calls the registered `Processor` (e.g. `LLMProcessor`) for each `OutputVar`.
5. **Rendering** — `TemplateRenderer.batch_render` applies the `output_schema` Jinja2 templates, substituting variable values. Simple `{{ var }}` references bypass Jinja2 to preserve non-string types (e.g. PIL Images).
6. **Atomic save** — The processed shard is written to `shard_<id>/` under `output_dir` using a temp-then-rename pattern with hostname + PID + UUID to avoid cross-host collisions on shared filesystems.
7. **State marker** — A `status.json` file is written to the state directory recording `success` or `failure` and the attempt count.

### SLURM mode

`launch_pipeline` generates and submits an sbatch array script. Each array task runs `shard_process.py` with `SLURM_ARRAY_TASK_ID` as the shard ID. The orchestrator polls job status via `squeue`, waits for the `settle_time_seconds`, checks `status.json` for each shard, and retries failed shards up to `max_retries`.

### Batch API mode (OpenAI, Anthropic)

The `BatchApiProcessor` delegates request submission to the batch orchestrator:
1. Requests are serialized to JSONL chunks respecting `max_chunk_bytes` / `max_requests_per_chunk`.
2. Each chunk is uploaded/submitted and a metadata receipt is written.
3. The mapper writes `__BATCH_SUBMITTED__:<custom_id>` placeholders into the output shards.
4. Results are later checked/collected via `mmirage.core.process.batch.status_checker` and `mmirage.core.process.batch.collector`.

---

## Key Design Decisions

### Registry pattern for loaders and processors
Both `DataLoaderRegistry` and `ProcessorRegistry` use a decorator-based registry. New loaders/processors self-register at import time, keeping the core pipeline agnostic of concrete implementations.

### Dacite + dataclasses for config
All configuration is expressed as plain Python dataclasses. `dacite` converts the raw YAML dict into typed objects, providing structural validation without a heavy schema library at runtime.

### JMESPath caching
Compiled JMESPath expressions are cached in a module-level dict to avoid recompilation on every sample — important for high-throughput processing.

### Atomic shard saves
Output shards are first written to a temporary directory with a host+PID+UUID suffix, then renamed. This guarantees crash-safe writes and avoids collisions on SLURM shared filesystems where multiple nodes may share a PID space.

### Separation of config and heavy deps
The `config/` package has minimal imports (no torch, sglang, transformers). The `core/process/processors/llm/config.py` module is also lightweight — it registers the processor configuration without importing the SGLang engine. The engine is only imported when a shard actually processes data, enabling fast CLI startup and documentation builds.

---

## See also

- [Pipeline](pipeline.md) — user-facing walkthrough of the data flow
- [Concepts](concepts.md) — vocabulary used throughout the codebase
- [Developer Guide](developer.md) — adding loaders and processors, running tests
