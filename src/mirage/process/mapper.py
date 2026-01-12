from typing import Dict, Any, List
from transformers import PreTrainedTokenizer

from mirage.config.variables import InputVar, OutputVar
from mirage.loader.extractor import VariableExtractor
from mirage.process.base import AutoProcessor, BaseProcessor, ProcessorConfig

import logging

from mirage.variables.environment import VariableEnvironment

logger = logging.getLogger(__name__)

class MIRAGEMapper():
    def __init__(self, 
                 processor_configs: List[ProcessorConfig], 
                input_vars: List[InputVar],
                output_vars: List[OutputVar]) -> None:
        self.processors: Dict[str, BaseProcessor] = dict()
        self.variable_extractor = VariableExtractor(input_vars)
        self.output_vars = output_vars

        for config in processor_configs:
            processor_cls = AutoProcessor.from_name(config.type)
            logger.info(f"✅ Sucessfully loaded processor of type {config.type}")

            self.processors[config.type] = processor_cls(config)


    def rewrite_batch(
            self,
            batch: Dict[str, List[Any]],
            ) -> List[VariableEnvironment]:
        batch_environment = self.variable_extractor.batch_extract_input_variables(batch)

        for output_var in self.output_vars:
            if output_var.type not in self.processors:
                raise RuntimeError(f"Output {output_var.type} not in registered processors: {self.processors.keys()}")

            processor = self.processors[output_var.type]
            batch_environment = processor.batch_process_sample(batch_environment, output_var)

        return batch_environment

