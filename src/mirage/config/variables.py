from dataclasses import dataclass
from typing import Literal, List, Dict, Any

@dataclass
class InputVar:
    name: str
    key: str


@dataclass
class OutputVar:
    name: str = ""
    type: str = ""
    output_type: Literal["plain", "JSON"] = "plain"

@dataclass
class ProcessingParams:
    inputs: List[InputVar]
    outputs: List[OutputVar]
    output_schema: Dict[str, Any]


