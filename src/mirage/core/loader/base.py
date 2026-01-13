from __future__ import annotations

import abc
from os import wait
from typing import Generic, Optional, Type, TypeVar
from dataclasses import dataclass

from datasets import Dataset

@dataclass
class BaseDataLoaderConfig:
    type: str


C = TypeVar("C", bound=BaseDataLoaderConfig)

class BaseDataLoader(abc.ABC, Generic[C]):

    @abc.abstractmethod
    def from_config(self, ds_config: C) -> Optional[Dataset]:
        ...

class DataLoaderRegistry:
    """
    Registry for managing and accessing available processors.

    Attributes:
        _registry (List[type]): List of registered processor classes.
    """

    _registry = dict()
    _config_registry = dict()
    
    @classmethod
    def register(cls, name: str, config_cls: Type[BaseDataLoaderConfig]):
        """
        Register a processor class.
        """
        def inner_register(clazz):
            cls._registry[name] = clazz
            cls._config_registry[name] = config_cls

        return inner_register

    
    @classmethod
    def get_processor(cls, name: str) -> Type[BaseDataLoader]:
        if name not in cls._registry:
            raise ValueError(f"Processor {name} not registered. Available processors are {list(cls._registry.keys())}")

        return cls._registry[name]

    @classmethod
    def get_config_cls(cls, name: str) -> Type[BaseDataLoaderConfig]:
        if name not in cls._config_registry:
            raise ValueError(f"Processor {name} not registered. Available processors are {list(cls._config_registry.keys())}")

        return cls._config_registry[name]


class AutoDataLoader:
    @classmethod
    def from_name(cls, name: str):
        """
        Determine and return the appropriate processor for the given file.

        Args:
            file (FileDescriptor): The file descriptor to process.

        Returns:
            Processor: The appropriate processor for the file, or None if no processor is found.
        """

        return DataLoaderRegistry.get_processor(name)

