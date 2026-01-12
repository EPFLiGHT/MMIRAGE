from __future__ import annotations

from typing import List, Dict, Any
from jmespath import search
from mirage.variables.environment import VariableEnvironment

from mirage.config.variables import InputVar

class VariableExtractor:
    def __init__(self, input_vars: List[InputVar]) -> None:
        self.input_vars = input_vars

    def extract_input_variables(
        self, sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract input variables from a dataset sample using JMESPath queries."""

        ret: Dict[str, Any] = {}
        for input_var in self.input_vars:
            value = search(input_var.key, sample)
            if value is None:
                raise ValueError(
                    f"Input variable '{input_var.name}' with key '{input_var.key}' "
                    "not found in the sample."
                )
            ret[input_var.name] = value

        return ret

    def batch_extract_input_variables(
            self, 
            batch: Dict[str, List[Any]],
        ) -> List[VariableEnvironment]:
        vars_samples: List[VariableEnvironment] = []  # input vars for each example

        # turn the dictionary of lists into a list of dictionaries
        batch_size = len(next(iter(batch.values())))
        batch_list: List[Dict[str, Any]] = [
            {k: batch[k][i] for k in batch.keys()} for i in range(batch_size)
        ]

        for sample in batch_list:
            current_vars = self.extract_input_variables(sample)
            vars_samples.append(VariableEnvironment(current_vars))


        return vars_samples

