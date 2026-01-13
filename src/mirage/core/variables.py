from __future__ import annotations

from typing import Dict, Any, List
from types import MappingProxyType
from jmespath import search

from mirage.core.process.variables import InputVar

class VariableEnvironment():
    def __init__(self, var_env: Dict[str, Any]) -> None:
        self._vars_env = var_env
    
    def with_variable(self, key: str, value: Any) -> VariableEnvironment:
        return VariableEnvironment(self._vars_env | {key : value})

    def to_dict(self) -> MappingProxyType:
        return MappingProxyType(self._vars_env)

    
    @staticmethod
    def from_input_variables(
        sample: Dict[str, Any],
        input_vars: List[InputVar]
    ) -> VariableEnvironment:
        """Extract input variables from a dataset sample using JMESPath queries."""

        ret: Dict[str, Any] = {}
        for input_var in input_vars:
            value = search(input_var.key, sample)
            if value is None:
                raise ValueError(
                    f"Input variable '{input_var.name}' with key '{input_var.key}' "
                    "not found in the sample."
                )
            ret[input_var.name] = value

        return VariableEnvironment(ret)
    
    @staticmethod
    def from_batch_input_variables(
            batch: Dict[str, List[Any]],
            input_vars: List[InputVar]
        ) -> List[VariableEnvironment]:
        vars_samples: List[VariableEnvironment] = []  # input vars for each example

        # turn the dictionary of lists into a list of dictionaries
        batch_size = len(next(iter(batch.values())))
        batch_list: List[Dict[str, Any]] = [
            {k: batch[k][i] for k in batch.keys()} for i in range(batch_size)
        ]

        for sample in batch_list:
            current_vars = VariableEnvironment.from_input_variables(sample, input_vars)
            vars_samples.append(current_vars)


        return vars_samples


