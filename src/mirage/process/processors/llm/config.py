from dataclasses import dataclass, field

from mirage.config.variables import InputVar, OutputVar
from typing import Dict, Optional, Type, Any, List
from pydantic import BaseModel, create_model

@dataclass
class LLMOutputVar(OutputVar):
    prompt: str = ""
    output_schema: List[str] = field(
        default_factory=list
    )  # empty list if output_type is "plain"

    def get_output_schema(self) -> Optional[Type[BaseModel]]:
        if self.output_type == "JSON" and self.output_schema:
            fields: Dict[str, Any] = {
                var: (str, ...) for var in self.output_schema
            }  # ... means required field
            return create_model(f"OutputSchema", **fields)
        return None


