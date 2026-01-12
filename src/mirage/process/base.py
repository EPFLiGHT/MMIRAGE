import abc
from typing import Dict, Any, List, Type

from mirage.config import OutputVar
from mirage.variables.environment import VariableEnvironment

class ProcessorConfig:
    type: str = ""


class BaseProcessor(abc.ABC):
    def __init__(self, config: ProcessorConfig) -> None:
        super().__init__()

    @abc.abstractmethod
    def batch_process_sample(self, batch: List[VariableEnvironment], output_var: OutputVar) -> List[VariableEnvironment]:
        ...


class ProcessorRegistry:
    """
    Registry for managing and accessing available processors.

    Attributes:
        _registry (List[type]): List of registered processor classes.
    """

    _registry = dict()
    
    @classmethod
    def register(cls, name: str):
        """
        Register a processor class.
        """
        def inner_register(clazz):
            cls._registry[name] = clazz

        return inner_register

    
    @classmethod
    def get_processor(cls, name: str) -> Type[BaseProcessor]:
        if name not in cls._registry:
            raise ValueError(f"Processor {name} not registered. Available processors are {cls._registry.keys()}")

        return cls._registry[name]


class AutoProcessor:
    @classmethod
    def from_name(cls, name: str):
        """
        Determine and return the appropriate processor for the given file.

        Args:
            file (FileDescriptor): The file descriptor to process.

        Returns:
            Processor: The appropriate processor for the file, or None if no processor is found.
        """

        return ProcessorRegistry.get_processor(name)
