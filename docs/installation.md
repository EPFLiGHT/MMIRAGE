# Installation

## Prerequisites

- Python 3.10 or later
- [PyTorch](https://pytorch.org/get-started/locally/) (recommended, for GPU acceleration)
- [SGLang](https://github.com/sgl-project/sglang) (required for LLM inference)

It is strongly recommended to install PyTorch and SGLang **before** installing MMIRAGE so that the correct GPU-enabled variants are resolved.

## From Source (recommended)

Clone the repository and install in editable mode:

```bash
git clone git@github.com:EPFLiGHT/MMIRAGE.git
pip install -e ./MMIRAGE
```

## Environment Setup

Several CLI features rely on environment variables (e.g. HuggingFace token, project paths). A helper script is provided to generate a `.env` file:

```bash
./scripts/generate_env.sh
```

## Optional Dependencies

Development tools (linters, type-checkers, testing):

```bash
pip install -e "./MMIRAGE[dev]"
```

## Verifying the Installation

```bash
mmirage --help
```
