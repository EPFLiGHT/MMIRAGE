"""LLM processor implementation using SGLang."""

from typing import List, override

import jinja2
from mmirage.core.process.base import BaseProcessor, ProcessorRegistry
from transformers import AutoTokenizer
from mmirage.core.process.variables import VariableEnvironment
import sglang as sgl
import json
from dataclasses import asdict

from mmirage.core.process.processors.llm.config import LLMOutputVar, SGLangLLMConfig


@ProcessorRegistry.register("llm", SGLangLLMConfig, LLMOutputVar)
class LLMProcessor(BaseProcessor[LLMOutputVar]):
    """LLM processor for generating text using SGLang.

    Supports both plain text and JSON output formats, with automatic
    chat template formatting and structured output validation.

    Attributes:
        llm: SGLang engine for text generation.
        tokenizer: Hugging Face tokenizer for chat template formatting.
        sampling_params: Default sampling parameters for generation.
    """

    def __init__(self, engine_args: SGLangLLMConfig, **kwargs) -> None:
        """Initialize the LLM processor.

        Args:
            engine_args: Configuration for SGLang server and sampling parameters.
            **kwargs: Additional arguments passed to base class.
        """
        super().__init__(engine_args, **kwargs)
        self.llm = sgl.Engine(**asdict(engine_args.server_args))
        self.tokenizer = AutoTokenizer.from_pretrained(
            engine_args.server_args.model_path
        )
        self.sampling_params = engine_args.default_sampling_params

    def build_prompt(
        self, prompt_template: str, vars_samples: List[VariableEnvironment]
    ) -> List[str]:
        """Build formatted prompts from a Jinja2 template and variable environments.

        Args:
            prompt_template: Jinja2 template string for the prompt.
            vars_samples: List of variable environments containing values.

        Returns:
            List of formatted prompts with chat template applied.
        """
        prompts_for_output = []

        jinja_template = jinja2.Template(prompt_template)

        for var in vars_samples:
            user_prompt = [
                {"role": "user", "content": jinja_template.render(**var.to_dict())}
            ]
            formatted_conv = self.tokenizer.apply_chat_template(
                user_prompt, tokenize=False, add_generation_prompt=True
            )
            prompts_for_output.append(formatted_conv)

        return prompts_for_output

    @override
    def batch_process_sample(
        self, batch: List[VariableEnvironment], output_var: LLMOutputVar
    ) -> List[VariableEnvironment]:
        """Process a batch of variable environments to generate LLM outputs.

        Args:
            batch: List of variable environments to process.
            output_var: Output variable defining prompt and output format.

        Returns:
            List of updated variable environments with LLM-generated values.

        Raises:
            ValueError: If output_type is JSON but no output_schema is defined.
            RuntimeError: If output batch size doesn't match input batch size.
        """
        prompts_for_output = self.build_prompt(
            prompt_template=output_var.prompt,
            vars_samples=batch,
        )

        sampling_params_output = self.sampling_params.copy()

        if output_var.output_type == "JSON":
            json_schema = output_var.get_output_schema()
            if json_schema is None:
                raise ValueError(
                    f"Output variable {output_var.name} has output_type=JSON "
                    "but no output_schema defined."
                )

            sampling_params_output["json_schema"] = json.dumps(
                json_schema.model_json_schema()
            )

        outputs_for_output = self.llm.generate(
            prompts_for_output, sampling_params_output
        )

        if len(outputs_for_output) != len(batch):
            raise RuntimeError(
                f"Mismatch between the number of generated answers ({len(outputs_for_output)}) and the size of the batch ({len(batch)})"
            )

        mapped_batch = []
        for i, llm_output in enumerate(outputs_for_output):
            value = llm_output.get("text", "")
            if output_var.output_type == "JSON":
                value = json.loads(llm_output.get("text", "{}"))

            mapped_batch.append(batch[i].with_variable(output_var.name, value))

        return mapped_batch
