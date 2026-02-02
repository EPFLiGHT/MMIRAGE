"""Data loaders for MMIRAGE pipeline.

This module provides implementations for loading datasets from various sources:
- JSONL files
- Local Hugging Face datasets (saved to disk)

All loaders inherit from BaseDataLoader and are registered with DataLoaderRegistry
for dynamic instantiation based on configuration.
"""

from mmirage.core.loader.jsonl import JSONLDataConfig, JSONLDataLoader
from mmirage.core.loader.local_hf import LocalHFConfig, LocalHFDataLoader

__all__ = [
    "JSONLDataConfig",
    "JSONLDataLoader",
    "LocalHFDataLoader",
    "LocalHFConfig",
]
