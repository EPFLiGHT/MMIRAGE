# 🔧 Developer Guide

This page covers everything needed to contribute to or develop MMIRAGE locally.

For a user-facing overview of the pipeline and architecture, see
[Pipeline](pipeline.md) and [Architecture](architecture.md).

---

## Setting Up a Development Environment

Clone the repository and install all dependencies including dev tools:

```bash
git clone git@github.com:EPFLiGHT/MMIRAGE.git
cd MMIRAGE
pip install -e ".[dev]"
```

The `dev` extra installs:

| Tool | Purpose |
|---|---|
| `ruff` | Fast Python linter |
| `black` | Code formatter |
| `mypy` | Static type checker |
| `pytest` | Test runner |
| `ipykernel` | Jupyter kernel for exploratory notebooks |

---

## Running Tests

```bash
pytest tests/
```

Individual test files:

```bash
pytest tests/test_batch_chunking.py -v
pytest tests/test_integration_batch_pipeline.py -v
```

The test suite does **not** require a GPU or a live OpenAI key — heavy dependencies are monkeypatched in integration tests.

### Shell-based smoke tests

Two shell scripts run end-to-end pipeline smoke tests using mock data:

```bash
# Text-only pipeline
bash tests/test_mock_data.sh

# Vision pipeline
bash tests/test_mock_vision.sh
```

These tests require a functional GPU environment and a working SGLang install (i.e. the `gpu` extra).

---

## Code Style

Format code with Black:

```bash
black src/ tests/
```

Lint with Ruff:

```bash
ruff check src/ tests/
```

Type-check with mypy:

```bash
mypy src/mmirage/
```

---

## Project Structure Conventions

- **`config/`** — Pure dataclasses, minimal imports. No heavy ML libraries allowed.
- **`core/`** — Implementation. Heavy imports (torch, sglang) are allowed but must stay inside the module, not at the top of `config.py`.
- **`cli_utils/`** — Thin helpers used by the CLI only.
- **Registries** — New loaders/processors must self-register using `@DataLoaderRegistry.register` or `@ProcessorRegistry.register` at module import time.

---

## Adding a New Loader

1. Create a file under `src/mmirage/core/loader/`.
2. Define a `@dataclass` config inheriting from `BaseDataLoaderConfig` with `type = "your_type"`.
3. Implement `BaseDataLoader` and decorate the class with `@DataLoaderRegistry.register("your_type", YourConfig)`.
4. Import the new module in `src/mmirage/config/utils.py` under the "Register built-in processors/loaders" comment.

Example skeleton:

```python
from dataclasses import dataclass
from mmirage.core.loader.base import BaseDataLoader, BaseDataLoaderConfig, DataLoaderRegistry, DatasetLike

@dataclass
class MyLoaderConfig(BaseDataLoaderConfig):
    type: str = "mytype"
    path: str = ""

@DataLoaderRegistry.register("mytype", MyLoaderConfig)
class MyLoader(BaseDataLoader[MyLoaderConfig]):
    def from_config(self, ds_config: MyLoaderConfig) -> DatasetLike:
        ...
```

---

## Adding a New Processor

1. Create a directory under `src/mmirage/core/process/processors/yourname/`.
2. Define a config dataclass inheriting from `BaseProcessorConfig` and an `OutputVar` subclass.
3. Implement `BaseProcessor` and register with `@ProcessorRegistry.register("yourname")`.
4. Import the config module in `src/mmirage/config/utils.py`.

---

## Building the Documentation

Documentation is built with Sphinx from the `docs/` directory.

Install documentation build dependencies:

```bash
pip install -r docs/requirements.txt
```

Build HTML output:

```bash
cd docs
make html
```

The output is written to `docs/_build/html/`. Open `docs/_build/html/index.html` in a browser to preview.

To clean and rebuild from scratch:

```bash
cd docs
make clean html
```

---

## Environment Variables Reference

| Variable | Used by | Description |
|---|---|---|
| `HF_TOKEN` | HuggingFace hub | Access token for gated/private models |
| `HF_HOME` | HuggingFace hub | Local cache directory |
| `SLURM_ARRAY_TASK_ID` | `LoadingParams` | Resolved as `shard_id` in SLURM jobs |
| `SLURM_GPUS_ON_NODE` | `SGLangServerArgs` | Auto-populates `tp_size` when not set |
| `MMIRAGE_COLLECT_STATS` | `shard_process` | Set to `1` to enable GPU/throughput benchmarking |
| `PYTHONPATH` | SLURM scripts | Injected by the generated sbatch script to point at `src/` |

---

## Debugging Tips

### Enable debug logging

```bash
mmirage run --config configs/config_mock.yaml --log-level DEBUG
```

### Run a single shard locally without SLURM

```bash
mmirage run --config configs/config.yaml --shard-id 0
```

### Inspect shard state without running

```bash
mmirage check --config configs/config.yaml
```

The command prints a JSON summary of `total / successful / running / failed / max_retries_exceeded` shard counts and exits with `0` if all shards succeeded.

### Check the generated sbatch script

The `submit` command prints the sbatch script to logs at DEBUG level. Use `--log-level DEBUG` when debugging SLURM submission issues.

---

## Common Issues

| Symptom | Likely cause | Fix |
|---|---|---|
| `ImportError: No module named 'sglang'` | GPU extra not installed | `pip install -e ".[gpu]"` |
| `Only shard 0 runs locally` | `loading_params.shard_id: "$SLURM_ARRAY_TASK_ID"` falls back to `0` when the env var is unset | Set `loading_params.shard_id` explicitly, or run `mmirage run --shard-id N` when testing locally |
| `FileNotFoundError` for EDF env | `edf_env` path does not exist | Remove `edf_env` from config or fix the path |

---

## See also

- [Architecture](architecture.md) — internal module layout and design decisions
- [Pipeline](pipeline.md) — end-to-end data flow
- [Configuration Reference](configuration.md) — full parameter reference
- [CLI Reference](cli.md) — all subcommands and flags
