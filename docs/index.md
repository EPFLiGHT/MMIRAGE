# MMIRAGE Documentation

<p align="center">
  <img src="https://raw.githubusercontent.com/EPFLiGHT/MMIRAGE/main/mmirage_logo_with_text.png" alt="MMIRAGE logo" width="480"/>
</p>

**MMIRAGE** — **M**odular **M**ultimodal **I**ntelligent **R**eformatting and **A**ugmentation **G**eneration **E**ngine — is an advanced platform for large-scale dataset processing using generative models, including vision-language models (VLMs).

---

::::{grid} 2
:gutter: 3

:::{grid-item-card} 🚀 Getting Started
:link: installation
:link-type: doc
Install MMIRAGE and run your first pipeline in minutes.
:::

:::{grid-item-card} 📖 Configuration Reference
:link: configuration
:link-type: doc
Full YAML configuration reference for all parameters.
:::

:::{grid-item-card} 🖥️ CLI Reference
:link: cli
:link-type: doc
All `mmirage` subcommands, flags, and examples.
:::

:::{grid-item-card} 🏗️ Architecture
:link: architecture
:link-type: doc
Package layout, data flow, and design decisions.
:::

:::{grid-item-card} 🛠️ Developer Guide
:link: developer
:link-type: doc
Testing, linting, adding loaders/processors, and debugging.
:::

:::{grid-item-card} 📦 API Reference
:link: api/index
:link-type: doc
Auto-generated documentation for every public module.
:::

::::

## Key Features

- **Multimodal support** — process text and images with vision-language models.
- **YAML-driven** — configure every aspect of a pipeline via a single file using Jinja2 templating and JMESPath queries.
- **Scalable** — native sharding with multi-node SLURM support.
- **Modular** — pluggable processors, loaders, and writers.
- **Automatic retry** — configurable shard-level retry with budget tracking.
- **Structured output** — produce plain text or validated JSON.

```{toctree}
:maxdepth: 2
:hidden:
:caption: User Guide

installation
quickstart
configuration
cli
architecture
developer
```

```{toctree}
:maxdepth: 3
:hidden:
:caption: API Reference

api/index
```
