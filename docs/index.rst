.. _index:

MMIRAGE Documentation
=====================

.. image:: https://raw.githubusercontent.com/EPFLiGHT/MMIRAGE/main/mmirage_logo_with_text.png
   :alt: MMIRAGE logo
   :align: center
   :width: 480px

|

**MMIRAGE** — **M**\odular **M**\ultimodal **I**\ntelligent **R**\eformatting and **A**\ugmentation **G**\eneration **E**\ngine — is an advanced platform for large-scale dataset processing using generative models, including vision-language models (VLMs).

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: :octicon:`rocket` Getting started
      :link: installation
      :link-type: doc

      Install MMIRAGE and run your first pipeline in minutes.

   .. grid-item-card:: :octicon:`book` Configuration reference
      :link: configuration
      :link-type: doc

      Full YAML configuration reference for all parameters.

   .. grid-item-card:: :octicon:`terminal` CLI reference
      :link: cli
      :link-type: doc

      All ``mmirage`` subcommands, flags, and examples.

   .. grid-item-card:: :octicon:`code-square` API reference
      :link: api/index
      :link-type: doc

      Auto-generated documentation for every public module.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: User Guide

   installation
   quickstart
   configuration
   cli

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: API Reference

   api/index

Key features
------------

- **Multimodal support** — process text and images with vision-language models.
- **YAML-driven** — configure every aspect of a pipeline via a single file using Jinja2 templating and JMESPath queries.
- **Scalable** — native sharding with multi-node SLURM support.
- **Modular** — pluggable processors, loaders, and writers.
- **Automatic retry** — configurable shard-level retry with budget tracking.
- **Structured output** — produce plain text or validated JSON.
