"""Mapper for orchestrating variable transformations."""

from typing import Dict, Any, List, cast

from mirage.core.process.variables import BaseVar, InputVar, OutputVar
from mirage.core.process.base import AutoProcessor, BaseProcessor, BaseProcessorConfig

import logging

from mirage.core.process.variables import VariableEnvironment

logger = logging.getLogger(__name__)


class MIRAGEMapper:
    """Mapper for orchestrating variable transformations in the MIRAGE pipeline.

    Manages processors, validates variable dependencies, and applies
    transformations to batches of data.

    Attributes:
        processors: Dictionary mapping processor types to processor instances.
        output_vars: List of output variables to generate.
        input_vars: List of input variables to extract.
    """

    def __init__(
        self,
        processor_configs: List[BaseProcessorConfig],
        input_vars: List[InputVar],
        output_vars: List[OutputVar],
    ) -> None:
        """Initialize the MIRAGE mapper.

        Args:
            processor_configs: List of processor configurations.
            input_vars: List of input variable definitions.
            output_vars: List of output variable definitions.
        """
        self.processors: Dict[str, BaseProcessor] = dict()
        self.output_vars = output_vars
        self.input_vars = input_vars

        for config in processor_configs:
            processor_cls = AutoProcessor.from_name(config.type)
            logger.info(f"✅ Successfully loaded processor of type {config.type}")

            self.processors[config.type] = processor_cls(config)

    def validate_vars(self) -> bool:
        """Validate that all output variables are computable.

        Checks that each output variable can be computed given the
        available variables (inputs and previously computed outputs).

        Returns:
            True if all variables are computable, False otherwise.
        """
        vars = cast(List[BaseVar], self.input_vars.copy())

        for output_var in self.output_vars:
            if not output_var.is_computable(vars):
                context = list(map(lambda v: v.name, vars))
                logger.info(
                    f"⚠️ Variable {output_var.name} not computable given current context: {context}"
                )
                return False

            vars.append(output_var)

        return True

    def rewrite_batch(
        self,
        batch: Dict[str, List[Any]],
    ) -> List[VariableEnvironment]:
        """Transform a batch of samples by computing output variables.

        Args:
            batch: Dictionary mapping column names to lists of values.

        Returns:
            List of VariableEnvironments with all output variables computed.

        Raises:
            RuntimeError: If an output variable type has no registered processor.
        """
        batch_environment = VariableEnvironment.from_batch_input_variables(
            batch, self.input_vars
        )

        for output_var in self.output_vars:
            if output_var.type not in self.processors:
                raise RuntimeError(
                    f"Output {output_var.type} not in registered processors: {self.processors.keys()}"
                )

            processor = self.processors[output_var.type]
            batch_environment = processor.batch_process_sample(
                batch_environment, output_var
            )

        return batch_environment
