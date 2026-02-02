"""LLM processor implementation using SGLang with multimodal support."""

from typing import List, Tuple, Any, override
import sys
import json
import logging
from dataclasses import asdict

import jinja2
from transformers import AutoTokenizer
import sglang as sgl

from mmirage.core.process.base import BaseProcessor, ProcessorRegistry
from mmirage.core.process.variables import VariableEnvironment
from mmirage.core.process.processors.llm.config import LLMOutputVar, SGLangLLMConfig

logger = logging.getLogger(__name__)


@ProcessorRegistry.register("llm", SGLangLLMConfig, LLMOutputVar)
class LLMProcessor(BaseProcessor[LLMOutputVar]):
    """LLM processor for generating text using SGLang.

    Supports both plain text and JSON output formats, with automatic
    chat template formatting and structured output validation.
    Also supports multimodal (vision-language) inputs.

    Attributes:
        llm: SGLang engine for text generation.
        tokenizer: Hugging Face tokenizer for chat template formatting.
        sampling_params: Default sampling parameters for generation.
        chat_template: Optional chat template for VLM models.
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
        self.chat_template = engine_args.chat_template

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

    def build_multimodal_prompt(
        self, prompt_template: str, var_env: VariableEnvironment
    ) -> Tuple[str, List[Any]]:
        """Build a prompt and extract images for SGLang Engine.

        Args:
            prompt_template: Jinja2 template string for the prompt.
            var_env: Variable environment containing values and images.

        Returns:
            Tuple of (formatted_prompt, list_of_images).
        """
        jinja_template = jinja2.Template(prompt_template)
        formatted_prompt = jinja_template.render(**var_env.to_dict())
        images = var_env.get_images()
        return formatted_prompt, images

    def _get_image_token(self) -> str:
        """Get the image token for the current chat template.

        Returns:
            Image token string for VLM models.

        Raises:
            ValueError: If chat template is not configured for multimodal.
        """
        if not self.chat_template:
            # Try to use a default based on model
            return "<image>"

        # Import chat templates from sglang if available
        try:
            from sglang.srt.conversation import chat_templates

            if self.chat_template in chat_templates:
                conv = chat_templates[self.chat_template].copy()
                return conv.image_token
        except ImportError:
            pass

        # Common image tokens for known templates
        image_tokens = {
            "qwen2-vl": "<|vision_start|><|image_pad|><|vision_end|>",
            "llava": "<image>",
            "internvl": "<image>",
            "phi3_v": "<|image_1|>",
        }

        return image_tokens.get(self.chat_template, "<image>")

    @override
    def batch_process_sample(
        self, batch: List[VariableEnvironment], output_var: LLMOutputVar
    ) -> List[VariableEnvironment]:
        """Process a batch of variable environments to generate LLM outputs.

        Automatically handles text-only and multimodal samples separately
        for optimal batching efficiency.

        Args:
            batch: List of variable environments to process.
            output_var: Output variable defining prompt and output format.

        Returns:
            List of updated variable environments with LLM-generated values.

        Raises:
            ValueError: If output_type is JSON but no output_schema is defined.
            RuntimeError: If output batch size doesn't match input batch size.
        """
        nb_samples = len(batch)

        # Prepare sampling params
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

        # Separate samples into text-only and multimodal groups
        text_only_indices = []
        multimodal_indices = []

        for i in range(nb_samples):
            if batch[i].has_images():
                multimodal_indices.append(i)
            else:
                text_only_indices.append(i)

        # Initialize results
        results = [None] * nb_samples

        # Process text-only samples in batch if any exist
        if text_only_indices:
            text_only_envs = [batch[i] for i in text_only_indices]
            text_only_prompts = self.build_prompt(output_var.prompt, text_only_envs)

            try:
                text_only_outputs = self.llm.generate(
                    prompt=text_only_prompts,
                    sampling_params=sampling_params_output,
                )

                if not isinstance(text_only_outputs, list) or len(text_only_outputs) != len(text_only_indices):
                    raise RuntimeError(
                        f"Mismatch between text-only prompts and outputs for '{output_var.name}': "
                        f"{len(text_only_prompts)} vs "
                        f"{len(text_only_outputs) if isinstance(text_only_outputs, list) else 'non-list'}"
                    )

                for idx, i in enumerate(text_only_indices):
                    value = text_only_outputs[idx].get("text", "").strip()
                    if output_var.output_type == "JSON":
                        try:
                            value = json.loads(value)
                        except json.JSONDecodeError:
                            value = {}
                    results[i] = batch[i].with_variable(output_var.name, value)

            except Exception as e:
                logger.error(
                    f"Batch generation failed for text-only samples in output '{output_var.name}': {e}"
                )
                # On error, set empty values for failed samples
                for i in text_only_indices:
                    empty_val = {} if output_var.output_type == "JSON" else ""
                    results[i] = batch[i].with_variable(output_var.name, empty_val)

        # Process multimodal samples in batch if any exist
        if multimodal_indices:
            image_token = self._get_image_token()

            # Build prompts with image tokens
            jinja_template = jinja2.Template(output_var.prompt)
            multimodal_prompts = []
            multimodal_images = []

            for i in multimodal_indices:
                var_env = batch[i]
                base_prompt = jinja_template.render(**var_env.to_dict())

                # Format prompt with chat template
                user_prompt = [{"role": "user", "content": base_prompt}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    user_prompt, tokenize=False, add_generation_prompt=True
                )

                # Add image token
                formatted_prompt = formatted_prompt + f"\n{image_token}\n"
                multimodal_prompts.append(formatted_prompt)
                multimodal_images.append(var_env.get_images())

            try:
                multimodal_outputs = self.llm.generate(
                    prompt=multimodal_prompts,
                    sampling_params=sampling_params_output,
                    image_data=multimodal_images,
                )

                if not isinstance(multimodal_outputs, list) or len(multimodal_outputs) != len(multimodal_indices):
                    raise RuntimeError(
                        f"Mismatch between multimodal prompts and outputs for '{output_var.name}': "
                        f"{len(multimodal_prompts)} vs "
                        f"{len(multimodal_outputs) if isinstance(multimodal_outputs, list) else 'non-list'}"
                    )

                for idx, i in enumerate(multimodal_indices):
                    value = multimodal_outputs[idx].get("text", "").strip()
                    if output_var.output_type == "JSON":
                        try:
                            value = json.loads(value)
                        except json.JSONDecodeError:
                            value = {}
                    results[i] = batch[i].with_variable(output_var.name, value)

            except Exception as e:
                logger.error(
                    f"Batch generation failed for multimodal samples in output '{output_var.name}': {e}"
                )
                # On error, set empty values for failed samples
                for i in multimodal_indices:
                    empty_val = {} if output_var.output_type == "JSON" else ""
                    results[i] = batch[i].with_variable(output_var.name, empty_val)

        return results

    def shutdown(self):
        """Shutdown the LLM engine."""
        try:
            self.llm.shutdown()
        except Exception as e:
            logger.warning(f"Error shutting down LLM: {e}")
