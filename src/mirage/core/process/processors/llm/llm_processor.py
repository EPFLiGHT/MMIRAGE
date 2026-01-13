from typing import List, override
from mirage.core.process.base import BaseProcessor, ProcessorRegistry
from transformers import AutoTokenizer
from mirage.core.process.variables import VariableEnvironment
import sglang as sgl
import json
from dataclasses import asdict

from mirage.core.process.processors.llm.config import LLMOutputVar, SGLangLLMConfig


@ProcessorRegistry.register("llm", SGLangLLMConfig, LLMOutputVar)
class LLMProcessor(BaseProcessor[LLMOutputVar]):
    def __init__(self, engine_args: SGLangLLMConfig, **kwargs) -> None:
        super().__init__(engine_args, **kwargs)
        self.llm = sgl.Engine(**asdict(engine_args.server_args))
        self.tokenizer = AutoTokenizer.from_pretrained(engine_args.server_args.model_path)
        self.sampling_params = engine_args.default_sampling_params

    def build_prompt(
            self,
            prompt_template: str,
            vars_samples: List[VariableEnvironment]
        ) -> List[str]:
        prompts_for_output = []

        for var in vars_samples:
            user_prompt = [{
                "role" : "user", 
                "content" : prompt_template.format(**var.to_dict())
            }]
            formatted_conv = self.tokenizer.apply_chat_template(user_prompt, tokenize=False, add_generation_prompt=True)
            prompts_for_output.append(formatted_conv)

        return prompts_for_output

    
    @override
    def batch_process_sample(self, batch: List[VariableEnvironment], variable: LLMOutputVar) -> List[VariableEnvironment]:
        prompts_for_output = self.build_prompt(
                prompt_template=variable.prompt,
                vars_samples=batch,
        )

        sampling_params_output = self.sampling_params.copy()

        if variable.output_type == "JSON":
            json_schema = variable.get_output_schema()
            if json_schema is None:
                raise ValueError(
                    f"Output variable {variable.name} has output_type=JSON "
                    "but no output_schema defined."
                )

            sampling_params_output["json_schema"] = json.dumps(json_schema.model_json_schema())

        outputs_for_output = self.llm.generate(
            prompts_for_output, sampling_params_output
        )

        if len(outputs_for_output) != len(batch):
            raise RuntimeError(
                f"Mismatch between the number of generated answers ({len(outputs_for_output)}) and the size of the batch ({len(batch)})"
            )

        # Update the variables
        mapped_batch = []
        for i, llm_output in enumerate(outputs_for_output):
            value = llm_output.get("text", "")
            if variable.output_type == "JSON":
                value = json.loads(llm_output.get("text", ""))

            mapped_batch.append(batch[i].with_variable(variable.name, value))

        return mapped_batch




