from dataclasses import dataclass, field

import logging
from typing import Dict, Optional, Sequence, Type, Any, List
from pydantic import BaseModel, create_model

from mirage.core.process.variables import BaseVar, OutputVar
from sglang.srt.server_args import ServerArgs

from mirage.core.process.base import BaseProcessorConfig
from jinja2 import Environment, PackageLoader, meta

logger = logging.getLogger(__name__)
env = Environment()

@dataclass
class SGLangLLMConfig(BaseProcessorConfig):
    server_args: ServerArgs = field(default_factory=ServerArgs(model_path="none"))
    default_sampling_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LLMOutputVar(OutputVar):
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

    def is_computable(self, vars: Sequence[BaseVar]) -> bool:
        parsed_content = env.parse(self.prompt)
        template_vars = meta.find_undeclared_variables(parsed_content)

        var_names = set(map(lambda v : v.name, vars))
        undeclared_vars = template_vars - var_names

        if len(undeclared_vars) > 0:
            logger.info(f"⚠️ Undeclared variables found for {self.name}: {undeclared_vars}")
            return False
        
        return True


