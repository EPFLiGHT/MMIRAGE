from __future__ import annotations

from typing import Dict, Any 

from types import MappingProxyType

class VariableEnvironment():
    def __init__(self, var_env: Dict[str, Any]) -> None:
        self._vars_env = var_env
    
    def with_variable(self, key: str, value: Any) -> VariableEnvironment:
        return VariableEnvironment(self._vars_env | {key : value})

    def to_dict(self) -> MappingProxyType:
        return MappingProxyType(self._vars_env)

