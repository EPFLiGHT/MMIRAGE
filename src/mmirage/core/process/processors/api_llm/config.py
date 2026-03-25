"""Configuration for API-based LLM processor in MMIRAGE."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Type
from jinja2 import Environment, meta
from pydantic import BaseModel, create_model
from mmirage.core.process.base import BaseProcessorConfig
from mmirage.core.process.variables import BaseVar, OutputVar

logger = logging.getLogger(__name__)
env = Environment()

@dataclass
class APILLMConfig(BaseProcessorConfig):
    model: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    base_url: Optional[str] = None
    default_sampling_params: Dict[str, Any] = field(default_factory=dict)
    max_concurrency: int = 8
    max_retries: int = 3
    retry_base_delay: float = 1.0
    
@dataclass
class APILLMOutputVar(OutputVar):
    prompt: str = ""                                                                                                           
    output_schema: List[str] = field(default_factory=list)
    output_type: str = ""                                                                                                      
                  
    def get_output_schema(self) -> Optional[Type[BaseModel]]:
        if self.output_type == "JSON" and self.output_schema:
            fields: Dict[str, Any] = {var: (str, ...) for var in self.output_schema}
            return create_model("OutputSchema", **fields)
        return None

    def is_computable(self, vars: Sequence[BaseVar]) -> bool:                                                                  
        parsed_content = env.parse(self.prompt)
        template_vars = meta.find_undeclared_variables(parsed_content)                                                         
        var_names = set(map(lambda v: v.name, vars))
        undeclared_vars = template_vars - var_names                                                                            
        if undeclared_vars:
            logger.info(f"Undeclared variables found for {self.name}: {undeclared_vars}")                                   
            return False                                                                                                       
        return True