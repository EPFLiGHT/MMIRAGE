"""Processing module for MIRAGE pipeline.

This module provides the core processing infrastructure:
- Base classes for processors and variables
- MMIRAGEMapper for orchestrating transformations
- LLM processor implementation for generative tasks (including multimodal)

Processors are responsible for generating new output variables from
existing variables, enabling flexible data transformations.
"""

from mirage.core.process.processors.llm.config import LLMOutputVar, SGLangLLMConfig
from mirage.core.process.processors.llm.llm_processor import LLMProcessor

__all__ = ["LLMOutputVar", "SGLangLLMConfig", "LLMProcessor"]
