import abc
from dataclasses import dataclass
from typing import  Callable, Generic, List, Type, TypeVar

from mirage.core.process.variables import VariableEnvironment, OutputVar

@dataclass
class BaseProcessorConfig:
    type: str = ""


C = TypeVar("C", bound=OutputVar)

class BaseProcessor(abc.ABC, Generic[C]):
    def __init__(self, config: BaseProcessorConfig) -> None:
        super().__init__()
        self.config = config

    @abc.abstractmethod
    def batch_process_sample(self, batch: List[VariableEnvironment], output_var: C) -> List[VariableEnvironment]:
        ...

class ProcessorRegistry:
    """
    Registry for managing and accessing available processors.

    Attributes:
        _registry (dict[type]): List of registered processor classes.
        _config_registry (dict[type]): List of registered processor config classes.
        _output_var_registry (dict[type]): List of registered output var classes.
    """

    _registry = dict()
    _config_registry = dict()
    _output_var_registry = dict()
    
    @classmethod
    def register(cls, name: str, config_cls: Type[BaseProcessorConfig], output_var_cls: Type[OutputVar]) -> Callable:
        """
        Register a processor class.
        """
        def inner_register(clazz):
            cls._registry[name] = clazz
            cls._config_registry[name] = config_cls
            cls._output_var_registry[name] = output_var_cls

        return inner_register

    
    @classmethod
    def get_processor(cls, name: str) -> Type[BaseProcessor]:
        if name not in cls._registry:
            raise ValueError(f"Processor {name} not registered. Available processors are {list(cls._registry.keys())}")

        return cls._registry[name]

    @classmethod
    def get_config_cls(cls, name: str) -> Type[BaseProcessorConfig]:
        if name not in cls._config_registry:
            raise ValueError(f"Processor {name} not registered. Available processors are {list(cls._config_registry.keys())}")

        return cls._config_registry[name]

    @classmethod
    def get_output_var_cls(cls, name: str) -> Type[OutputVar]:
        if name not in cls._output_var_registry:
            raise ValueError(f"Processor {name} not registered. Available processors are {list(cls._output_var_registry.keys())}")

        return cls._output_var_registry[name]


class AutoProcessor:
    @classmethod
    def from_name(cls, name: str):
        """
        Retrieve the processor class registered under the given name.

        Args:
            name (str): The identifier of the processor to retrieve.

        Returns:
            Type[BaseProcessor]: The processor class associated with the given name.
        """

        return ProcessorRegistry.get_processor(name)


