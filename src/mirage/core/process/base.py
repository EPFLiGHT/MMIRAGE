"""Base classes and registry for processors in MIRAGE."""

import abc
from dataclasses import dataclass
from typing import  Callable, Generic, List, Type, TypeVar

from mirage.core.process.variables import VariableEnvironment, OutputVar

@dataclass
class BaseProcessorConfig:
    """Base configuration class for processors.

    All processor configurations must inherit from this class.

    Attributes:
        type: String identifier for the processor type (e.g., "llm").
    """
    type: str = ""


C = TypeVar("C", bound=OutputVar)

class BaseProcessor(abc.ABC, Generic[C]):
    """Abstract base class for data processors.

    Processors are responsible for transforming data by generating
    new output variables from existing variables.

    Type Parameters:
        C: The output variable type this processor works with.

    Attributes:
        config: Configuration object for this processor.
    """

    def __init__(self, config: BaseProcessorConfig) -> None:
        """Initialize the processor with configuration.

        Args:
            config: Configuration object for this processor.
        """
        super().__init__()
        self.config = config

    @abc.abstractmethod
    def batch_process_sample(self, batch: List[VariableEnvironment], output_var: C) -> List[VariableEnvironment]:
        """Process a batch of variable environments.

        Args:
            batch: List of variable environments to process.
            output_var: Output variable definition to generate.

        Returns:
            List of updated variable environments with the new output variable.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        ...

class ProcessorRegistry:
    """Registry for managing and accessing available processors.

    Provides a centralized registry for processor classes, their
    configuration classes, and their output variable classes.

    Attributes:
        _registry: Mapping from processor name to registered processor class.
        _config_registry: Mapping from processor name to its configuration class.
        _output_var_registry: Mapping from processor name to its output variable class.
    """

    _registry = dict()
    _config_registry = dict()
    _output_var_registry = dict()

    @classmethod
    def register(cls, name: str, config_cls: Type[BaseProcessorConfig], output_var_cls: Type[OutputVar]) -> Callable:
        """Register a processor class with its associated classes.

        Args:
            name: String identifier for the processor.
            config_cls: Configuration class associated with this processor.
            output_var_cls: Output variable class associated with this processor.

        Returns:
            Decorator function to register the processor class.
        """
        def inner_register(clazz):
            cls._registry[name] = clazz
            cls._config_registry[name] = config_cls
            cls._output_var_registry[name] = output_var_cls

        return inner_register


    @classmethod
    def get_processor(cls, name: str) -> Type[BaseProcessor]:
        """Get a registered processor class by name.

        Args:
            name: String identifier of the processor.

        Returns:
            The registered processor class.

        Raises:
            ValueError: If no processor is registered under the given name.
        """
        if name not in cls._registry:
            raise ValueError(f"Processor {name} not registered. Available processors are {list(cls._registry.keys())}")

        return cls._registry[name]

    @classmethod
    def get_config_cls(cls, name: str) -> Type[BaseProcessorConfig]:
        """Get a registered configuration class by processor name.

        Args:
            name: String identifier of the processor.

        Returns:
            The registered configuration class.

        Raises:
            ValueError: If no processor is registered under the given name.
        """
        if name not in cls._config_registry:
            raise ValueError(f"Processor {name} not registered. Available processors are {list(cls._config_registry.keys())}")

        return cls._config_registry[name]

    @classmethod
    def get_output_var_cls(cls, name: str) -> Type[OutputVar]:
        """Get a registered output variable class by processor name.

        Args:
            name: String identifier of the processor.

        Returns:
            The registered output variable class.

        Raises:
            ValueError: If no processor is registered under the given name.
        """
        if name not in cls._output_var_registry:
            raise ValueError(f"Processor {name} not registered. Available processors are {list(cls._output_var_registry.keys())}")

        return cls._output_var_cls[name]


class AutoProcessor:
    """Factory class for instantiating processors by name."""

    @classmethod
    def from_name(cls, name: str) -> Type[BaseProcessor]:
        """Retrieve a processor class by its registered name.

        Args:
            name: The registry name of the processor.

        Returns:
            The registered processor class.

        Raises:
            ValueError: If no processor is registered under the given name.
        """
        return ProcessorRegistry.get_processor(name)


