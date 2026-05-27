# Installation

## Prerequisites

- Python 3.12 or later
- An NVIDIA GPU with drivers installed (required for local SGLang-backed LLM inference; not needed for OpenAI Batch API mode)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (required only for GPU Docker usage)

## From Source (recommended)

Clone the repository and install the base package:

```bash
git clone git@github.com:EPFLiGHT/MMIRAGE.git
cd MMIRAGE
pip install -e .
```

The base install provides all core functionality except local GPU inference.

## GPU / Local Inference

To run pipelines locally using the SGLang-backed `llm` processor, install the `gpu` extra. It is strongly recommended to install [PyTorch](https://pytorch.org/get-started/locally/) matching your CUDA version **before** running this step:

```bash
pip install -e ".[gpu]"
```

This installs `sglang`, `sgl_kernel`, `xgrammar`, and `compressed_tensors`.

## Docker

Prebuilt images are published to GHCR for every push to `main`.

### GPU image

```bash
docker pull ghcr.io/epflight/mmirage:latest-gpu
docker run --rm -it --gpus all ghcr.io/epflight/mmirage:latest-gpu
```

The host must have NVIDIA GPU drivers, the NVIDIA Container Toolkit, and a recent Docker Engine with GPU support.

To build locally:

```bash
docker compose build mmirage
docker compose run --rm -it mmirage
```

### CPU image

Suitable for workflows that do not require a local GPU (e.g. OpenAI Batch API mode):

```bash
docker pull ghcr.io/epflight/mmirage:latest-cpu
docker run --rm -it ghcr.io/epflight/mmirage:latest-cpu
```

To build locally:

```bash
docker compose build mmirage-cpu
docker compose run --rm -it mmirage-cpu
```

## Environment Variables

Several features rely on environment variables (e.g. `HF_TOKEN` for private HuggingFace models, `SLURM_*` variables injected by the scheduler). A helper script generates a `.env` starter file:

```bash
./scripts/generate_env.sh
```

Key variables:

| Variable | Description |
|---|---|
| `HF_TOKEN` | HuggingFace API token for gated/private models |
| `HF_HOME` | HuggingFace cache directory (default: `~/hf`) |
| `SLURM_ARRAY_TASK_ID` | Shard ID injected automatically in SLURM array jobs |
| `SLURM_GPUS_ON_NODE` | Used to auto-detect `tp_size` for SGLang |
| `MMIRAGE_COLLECT_STATS` | Set to `1` to enable GPU/throughput benchmarking |

## Development Setup

Install linters, type checkers, and test dependencies:

```bash
pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest tests/
```

Format and lint the codebase:

```bash
black src/ tests/
ruff check src/ tests/
```

## Verifying the Installation

```bash
mmirage --help
```
