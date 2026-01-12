from dataclasses import dataclass
from typing import Any, Dict

from pydantic import BaseModel, create_model
from sglang.srt.server_args import ServerArgs

from mirage.config.variables import ProcessingParams
from mirage.config.generation import GenerationParams

# @dataclass 
# class EngineArgs:
#     server_args: ServerArgs
#     sampling_params: Dict[str, Any]

class ProcessorConfig:
    type: str = ""


@dataclass
class MirageConfig:
    # engine: EngineArgs
    processors: List[ProcessorConfig]
    processing_gen_params: GenerationParams
    processing_params: ProcessingParams
