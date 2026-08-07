# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from unittest.mock import MagicMock

# -- Path setup ----------------------------------------------------------------
# Point at the package source so autodoc can import mmirage without installing.
sys.path.insert(0, os.path.abspath("../src"))

# -- Lightweight datasets mock ------------------------------------------------
# The real `datasets` library imports pandas/pyarrow chains that may not be
# present in the docs build environment.  We pre-inject a minimal mock that
# exposes actual Python *classes* (not MagicMock instances) for Dataset and
# DatasetDict so that the PEP-604 union ``Dataset | DatasetDict`` in
# mmirage.core.loader.base works without a TypeError.


class _FakeDataset:
    """Stand-in for datasets.Dataset."""


class _FakeDatasetDict(dict):
    """Stand-in for datasets.DatasetDict."""


_datasets_mock = MagicMock()
_datasets_mock.Dataset = _FakeDataset
_datasets_mock.DatasetDict = _FakeDatasetDict
_datasets_mock.concatenate_datasets = MagicMock(return_value=_FakeDataset())
_datasets_mock.load_from_disk = MagicMock(return_value=_FakeDataset())

sys.modules["datasets"] = _datasets_mock
sys.modules["datasets.arrow_dataset"] = MagicMock()
sys.modules["datasets.dataset_dict"] = MagicMock()

# -- typing.override shim for Python < 3.12 ------------------------------------
# `override` was added to `typing` in Python 3.12.  The source uses it without
# a try/except in some files, so we inject a no-op shim before importing.
import typing as _typing

if not hasattr(_typing, "override"):

    def _override(f):  # type: ignore[return]
        return f

    _typing.override = _override  # type: ignore[attr-defined]

# -- Project information -------------------------------------------------------
project = "MMIRAGE"
copyright = "2026, Meditron team"
author = "Meditron team"
release = "0.1.4"

# -- General configuration -----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# MyST parser settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
]
myst_heading_anchors = 3

# -- Autodoc configuration -----------------------------------------------------
# Mock heavy runtime dependencies so autodoc can import the package without them.
autodoc_mock_imports = [
    # Heavy ML / inference libs
    "sglang",
    "transformers",
    "torch",
    # Async / server libs
    "pyzmq",
    "uvloop",
    "fastapi",
    "openai",
    "partial_json_parser",
    "sentencepiece",
    "sgl_kernel",
    "compressed_tensors",
    "msgspec",
    "nest_asyncio",
    "xgrammar",
    # Data / serialization (datasets is pre-mocked via sys.modules above)
    "datasets",
    "pyarrow",
    "fsspec",
    "dacite",
    "pydantic",
    "json_repair",
    # Utilities present in pyproject but may be absent locally
    "jmespath",
    "jinja2",
    "PIL",
    "yaml",
    "numpy",
    "huggingface_hub",
    "humanize",
    "tqdm",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}

autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_attr_annotations = True

# Suppress noisy-but-benign warnings:
#  - duplicate member descriptions caused by __init__.py re-exports
#  - unresolvable forward refs in mocked type annotations
#  - autodoc.import_object: modules that cannot be imported in doc env
suppress_warnings = [
    "ref.duplicate",
    "sphinx_autodoc_typehints.forward_reference",
    "myst.header",
    "autodoc",
]

# -- Intersphinx ---------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# -- HTML output ---------------------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_title = "MMIRAGE"
html_logo = "_static/logo.svg"

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/EPFLiGHT/MMIRAGE",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0"
                     viewBox="0 0 16 16" height="1em" width="1em"
                     xmlns="http://www.w3.org/2000/svg">
                  <path fill-rule="evenodd"
                        d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
                           0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13
                           -.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87
                           2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95
                           0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21
                           2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04
                           2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82
                           2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01
                           1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z">
                  </path>
                </svg>
            """,
            "class": "",
        },
    ],
}
