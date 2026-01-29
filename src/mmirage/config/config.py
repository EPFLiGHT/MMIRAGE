"""Configuration dataclasses for MMIRAGE pipeline."""

from dataclasses import dataclass
from typing import Any, Dict, List

from mmirage.config.loading import LoadingParams
from mmirage.core.process.base import BaseProcessorConfig
from mmirage.core.process.variables import InputVar, OutputVar


@dataclass
class ProcessingParams:
    """Parameters for processing dataset samples.

    Defines how input variables are extracted, outputs are generated,
    and the final output schema is constructed.

    Attributes:
        inputs: List of input variables to extract from source datasets.
        outputs: List of output variables to generate using processors.
        output_schema: Dictionary defining the structure of output samples.
        remove_columns: If True, removes all columns from original dataset.
    """

    inputs: List[InputVar]
    outputs: List[OutputVar]
    output_schema: Dict[str, Any]
    remove_columns: bool = False


@dataclass
class MMirageConfig:
    """Main configuration class for MMIRAGE pipeline.

    Contains all configuration needed to run a MMIRAGE processing pipeline,
    including processor configurations, dataset loading parameters, and
    processing parameters.

    Attributes:
        processors: List of processor configurations for data transformation.
        loading_params: Parameters for loading input datasets.
        processing_params: Parameters for processing dataset samples.
    """

    processors: List[BaseProcessorConfig]
    loading_params: LoadingParams
    processing_params: ProcessingParams
