"""Configuration module for MIRAGE pipeline.

This module provides configuration dataclasses and utilities for loading
and validating MIRAGE pipeline configurations.
"""

from mirage.config.config import MMirageConfig, ProcessingParams
from mirage.config.loading import LoadingParams

__all__ = [
    "MMirageConfig",
    "ProcessingParams",
    "LoadingParams",
]
