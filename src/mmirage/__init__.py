"""MMIRAGE - Modular Multimodal Intelligent Reformatting and Augmentation Generation Engine.

A platform for processing datasets using generative models including
vision-language models (VLMs).
"""
from __future__ import annotations

__version__ = "0.1.4"

from mmirage.config.config import MMirageConfig, ProcessingParams
from mmirage.config.loading import LoadingParams
from mmirage.config.utils import load_mmirage_config

__all__ = ["MMirageConfig", "ProcessingParams", "LoadingParams", "load_mmirage_config", "__version__"]
