from dataclasses import dataclass
from typing import Any, Dict, List 

from mirage.config.loading import LoadingParams
from mirage.core.process.base import BaseProcessorConfig 
from mirage.core.process.variables import InputVar, Variable

@dataclass
class ProcessingParams:
    inputs: List[InputVar]
    outputs: List[Variable]
    output_schema: Dict[str, Any]
    remove_columns: bool


@dataclass
class MirageConfig:
    processors: List[BaseProcessorConfig]
    loading_params: LoadingParams
    processing_params: ProcessingParams


