from dataclasses import dataclass
from typing import List, Dict, Any, Type

@dataclass
class InputVar:
    name: str = ""
    key: str = ""

@dataclass
class OutputVar:
    name: str = ""
    type: str = ""

@dataclass
class ProcessingParams:
    inputs: List[InputVar]
    outputs: List[OutputVar]
    output_schema: Dict[str, Any]


class OutputVarRegistry:
    """
    Registry for managing and accessing available processors.

    Attributes:
        _registry (List[type]): List of registered processor classes.
    """

    _registry = dict()
    
    @classmethod
    def register(cls, name: str, ):
        """
        Register a processor class.
        """
        def inner_register(config_cls: Type[OutputVar]):
            cls._registry[name] = config_cls

        return inner_register
    
    @classmethod
    def get_config(cls, name: str) -> Type[OutputVar]:
        if name not in OutputVarRegistry._registry:
            raise ValueError(f"OutputVar {name} not found in registry, available keys are: {list(OutputVarRegistry._registry.keys())}")

        return OutputVarRegistry._registry[name]


