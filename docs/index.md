# MMIRAGE Documentation

```{image} _static/logo.svg
:alt: MMIRAGE logo
:align: center
:width: 480px
```

MMIRAGE is an open-source platform for large-scale dataset processing using large language models (LLMs) and vision-language models (VLMs).

It provides a declarative, YAML-driven pipeline to extract variables from data samples, construct prompts, run model inference, render structured outputs, and distribute the work across shards — locally or on HPC clusters via SLURM.

---

## What is MMIRAGE?

MMIRAGE (**M**odular **M**ultimodal **I**ntelligent **R**eformatting and **A**ugmentation **G**eneration **E**ngine) lets you transform large datasets using generative models by:

- extracting named variables from each sample with JMESPath queries
- constructing prompts with Jinja2 templates
- running inference locally via a SGLang engine, or asynchronously via the OpenAI Batch API
- rendering processed outputs into any JSON schema you define
- splitting work across shards with automatic resume, retry, and merge

It natively supports text and image inputs, SLURM-based cluster execution, throughput benchmarking, and atomic crash-safe writes.

---

## Where to start

Depending on what you want to do, start in different places:

- to install MMIRAGE, read [Installation](installation.md)
- to run a first pipeline end to end, read [Quickstart](quickstart.md)
- to understand core concepts and terminology, read [Concepts](concepts.md)
- to understand the full pipeline data flow, read [Pipeline](pipeline.md)
- to process images with a VLM, read [Multimodal Processing](multimodal.md)
- to run at scale on a cluster, read [SLURM & Cluster Deployment](slurm.md)
- to use the OpenAI Batch API, read [Batch API](batch_api.md)
- to measure throughput and GPU efficiency, read [Benchmarking](benchmarking.md)
- to configure every parameter, read [Configuration Reference](configuration.md)
- to contribute to the codebase, read [Developer Guide](developer.md)

---

## Documentation map

::::{grid} 2
:gutter: 3

:::{grid-item-card} 📦 Installation
:link: installation
:link-type: doc
Set up MMIRAGE and prepare your environment.
:::

:::{grid-item-card} 🚀 Quickstart
:link: quickstart
:link-type: doc
Run a first pipeline end to end in minutes.
:::

:::{grid-item-card} 💡 Concepts
:link: concepts
:link-type: doc
Core vocabulary: shards, variables, schemas, execution modes.
:::

:::{grid-item-card} 🔄 Pipeline
:link: pipeline
:link-type: doc
Step-by-step walkthrough of what MMIRAGE does with your data.
:::

:::{grid-item-card} 🖼️ Multimodal Processing
:link: multimodal
:link-type: doc
Running VLMs on image datasets.
:::

:::{grid-item-card} 🔀 SLURM & Cluster Deployment
:link: slurm
:link-type: doc
Scaling pipelines across HPC nodes with SLURM.
:::

:::{grid-item-card} 🗂️ Batch API
:link: batch_api
:link-type: doc
Async inference via the OpenAI Batch API.
:::

:::{grid-item-card} 📊 Benchmarking
:link: benchmarking
:link-type: doc
Measuring throughput, GPU utilization, and efficiency.
:::

:::{grid-item-card} ⚙️ Configuration Reference
:link: configuration
:link-type: doc
Full YAML parameter reference for every section.
:::

:::{grid-item-card} 💻 CLI Reference
:link: cli
:link-type: doc
All `mmirage` subcommands, flags, and examples.
:::

:::{grid-item-card} 🏗️ Architecture
:link: architecture
:link-type: doc
Internal module layout and design decisions.
:::

:::{grid-item-card} 🔧 Developer Guide
:link: developer
:link-type: doc
Testing, code style, extending MMIRAGE, and debugging.
:::

::::

---

## Page guide

- [Installation](installation.md): set up MMIRAGE and prepare your environment
- [Quickstart](quickstart.md): run a first minimal pipeline from scratch
- [Concepts](concepts.md): learn the vocabulary used throughout the documentation
- [Pipeline](pipeline.md): understand what happens at each stage of the pipeline
- [Multimodal Processing](multimodal.md): configure image inputs and VLM chat templates
- [SLURM & Cluster Deployment](slurm.md): submit, monitor, and retry jobs on HPC clusters
- [Batch API](batch_api.md): send requests asynchronously to the OpenAI Batch API
- [Benchmarking](benchmarking.md): collect and interpret throughput and GPU efficiency metrics
- [Configuration Reference](configuration.md): complete reference for every YAML parameter
- [CLI Reference](cli.md): all subcommands, flags, and their behaviour
- [Architecture](architecture.md): internal package layout and key design decisions
- [Developer Guide](developer.md): contribute code, run tests, add loaders or processors

```{toctree}
:maxdepth: 1
:hidden:
:caption: Getting Started

installation
quickstart
concepts
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Pipeline & Processing

pipeline
multimodal
slurm
batch_api
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Reference

configuration
cli
benchmarking
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Developer

architecture
developer
```

```{toctree}
:maxdepth: 3
:hidden:
:caption: API Reference

api/index
```
