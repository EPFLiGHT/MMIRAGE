from dataclasses import dataclass, field

from mirage.core.process.base import Variable
from typing import Dict, Optional, Type, Any, List
from pydantic import BaseModel, create_model

from sglang.srt.server_args import ServerArgs

from mirage.core.process.base import BaseProcessorConfig

@dataclass
class SGLangLLMConfig(BaseProcessorConfig):
    server_args: ServerArgs = field(default_factory=ServerArgs(model_path="none"))
    default_sampling_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LLMOutputVar(Variable):
    prompt: str = ""
    output_schema: List[str] = field(
        default_factory=list
    )  # empty list if output_type is "plain"
    output_type: str = ""

    def get_output_schema(self) -> Optional[Type[BaseModel]]:
        if self.output_type == "JSON" and self.output_schema:
            fields: Dict[str, Any] = {
                var: (str, ...) for var in self.output_schema
            }  # ... means required field
            return create_model(f"OutputSchema", **fields)
        return None


