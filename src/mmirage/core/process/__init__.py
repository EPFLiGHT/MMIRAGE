"""Processing module for MMIRAGE pipeline.

This module provides the core processing infrastructure:
- Base classes for processors and variables
- MMIRAGEMapper for orchestrating transformations
- LLM processor implementation for generative tasks (including multimodal)

Processors are responsible for generating new output variables from
existing variables, enabling flexible data transformations.
"""

from mmirage.core.process.processors.llm.config import LLMOutputVar, SGLangLLMConfig
from mmirage.core.process.processors.llm.llm_processor import LLMProcessor
from mmirage.core.process.structured_outputs import (
    CompiledStructuredTask,
    DspyBackend,
    StructuredOutputBackend,
    StructuredTaskRunner,
    TaskSpecAST,
    compile_task,
    parse_task_spec,
    parse_task_spec_yaml,
)

__all__ = [
    "LLMOutputVar",
    "SGLangLLMConfig",
    "LLMProcessor",
    "TaskSpecAST",
    "StructuredOutputBackend",
    "DspyBackend",
    "CompiledStructuredTask",
    "parse_task_spec",
    "parse_task_spec_yaml",
    "compile_task",
    "StructuredTaskRunner",
]
