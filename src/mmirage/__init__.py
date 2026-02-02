"""MMIRAGE - Modular Multimodal Intelligent Reformatting and Augmentation Generation Engine.

A platform for processing datasets using generative models including
vision-language models (VLMs).
"""

__version__ = "0.2.0"

from mmirage.config import MMirageConfig, ProcessingParams, LoadingParams
from mmirage.config.utils import load_mmirage_config

__all__ = [
    "MMirageConfig",
    "ProcessingParams", 
    "LoadingParams",
    "load_mmirage_config",
    "__version__",
]
