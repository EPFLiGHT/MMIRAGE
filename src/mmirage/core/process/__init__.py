"""Processing module for MMIRAGE pipeline.

This module provides the core processing infrastructure:
- Base classes for processors and variables
- MMIRAGEMapper for orchestrating transformations
- LLM processor implementation for generative tasks (including multimodal)

Processors are responsible for generating new output variables from
existing variables, enabling flexible data transformations.
"""

try:
    from mmirage.core.process.processors.llm.config import LLMOutputVar, SGLangLLMConfig
    from mmirage.core.process.processors.llm.llm_processor import LLMProcessor
except ImportError:
    pass

from mmirage.core.process.processors.api_llm.config import APILLMConfig, APILLMOutputVar
from mmirage.core.process.processors.api_llm.api_llm_processor import APILLMProcessor

__all__ = ["LLMOutputVar", "SGLangLLMConfig", "LLMProcessor", "APILLMConfig", "APILLMOutputVar", "APILLMProcessor"]
