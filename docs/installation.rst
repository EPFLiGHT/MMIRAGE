.. _installation:

Installation
============

Prerequisites
-------------

- Python 3.10 or later
- `PyTorch <https://pytorch.org/get-started/locally/>`_ (recommended, for GPU acceleration)
- `SGLang <https://github.com/sgl-project/sglang>`_ (required for LLM inference)

It is strongly recommended to install PyTorch and SGLang **before** installing MMIRAGE so that the correct GPU-enabled variants are used.

From source (recommended)
--------------------------

Clone the repository and install in editable mode:

.. code-block:: bash

   git clone git@github.com:EPFLiGHT/MMIRAGE.git
   pip install -e ./MMIRAGE

Environment setup
-----------------

Several CLI features rely on environment variables (e.g. HuggingFace token, project paths). A helper script is provided to generate a ``.env`` file:

.. code-block:: bash

   ./scripts/generate_env.sh

Optional dependencies
---------------------

Development tools (linters, type-checkers, testing):

.. code-block:: bash

   pip install -e "./MMIRAGE[dev]"

Verifying the installation
--------------------------

.. code-block:: bash

   mmirage --help
