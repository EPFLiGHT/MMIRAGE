"""LLM processor implementation using SGLang with multimodal support."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid

import jinja2
try:
    import sglang as sgl
    SGLANG_AVAILABLE = True
except ImportError:
    SGLANG_AVAILABLE = False
from transformers import AutoTokenizer

from mmirage.core.process.base import BaseProcessor, ProcessorRegistry, TokenCounts
from mmirage.core.process.batch.orchestrator import BatchSubmissionOrchestrator
from mmirage.core.process.batch.registry import BatchAdapterFactory
from mmirage.core.process.processors.llm.config import LLMOutputVar, SGLangLLMConfig
from mmirage.core.process.variables import VariableEnvironment

try:
    from typing import override  # Python 3.12+
except ImportError:  # pragma: no cover
    from typing_extensions import override  # type: ignore


logger = logging.getLogger(__name__)

# Common image tokens for known templates
IMAGE_TOKENS = {
    "qwen2-vl": "<|vision_start|><|image_pad|><|vision_end|>",
    "llava": "<image>",
    "internvl": "<image>",
    "phi3_v": "<|image_1|>",
}


@ProcessorRegistry.register("llm", SGLangLLMConfig, LLMOutputVar)
class LLMProcessor(BaseProcessor[LLMOutputVar]):
    """LLM processor for generating text using SGLang.

    Supports both plain text and JSON output formats, with automatic
    chat template formatting and structured output validation.

    Also supports multimodal (vision-language) inputs. For SGLang, `image_data`
    is expected to be *aligned with prompts*: a list where each element is
    either None (text-only), a single image (path/URL/PIL), or (optionally)
    a list of images for that prompt.

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

        batch_provider_cfg = engine_args.batch_provider
        is_provider_batch_enabled = bool(batch_provider_cfg and batch_provider_cfg.enabled)
        self._model_load_seconds: float = 0.0

        # In provider-batch mode we only build payloads/metadata and should not
        # initialize GPU-backed SGLang runtime.
        if is_provider_batch_enabled:
            self.llm = None
            self.tokenizer = None
        else:
            if not SGLANG_AVAILABLE:
                raise RuntimeError(
                    "SGLang is not installed. Install with: pip install 'mmirage[gpu]' "
                    "or, from a source checkout, pip install -e '.[gpu]'"
                )

            server_kwargs = asdict(engine_args.server_args)
            extra = server_kwargs.pop("extra_engine_args", {}) or {}
            server_kwargs.update(extra)
            _load_start = time.monotonic()
            self.llm = sgl.Engine(**server_kwargs)
            self._model_load_seconds = time.monotonic() - _load_start
            self.tokenizer = AutoTokenizer.from_pretrained(
                engine_args.server_args.model_path,
                trust_remote_code=getattr(engine_args.server_args, "trust_remote_code", False),
            )

        self.sampling_params = engine_args.default_sampling_params
        self.chat_template = engine_args.chat_template
        self._batch_adapter = None
        self._batch_provider_config = None
        self._text_orchestrator: Optional[BatchSubmissionOrchestrator] = None
        self._multimodal_orchestrator: Optional[BatchSubmissionOrchestrator] = None
        self._batch_request_counter = 0
        self._global_row_offset = 0
        self._setup_batch_runtime()

        # Cumulative token counts across all generate() calls in this processor's lifetime.
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0

    def _setup_batch_runtime(self) -> None:
        provider_cfg = self.config.batch_provider
        if provider_cfg is None:
            return

        if not provider_cfg.enabled:
            return

        self._batch_provider_config = provider_cfg
        self._batch_adapter = BatchAdapterFactory.from_config(provider_cfg)
        run_id = uuid.uuid4().hex[:6]

        self._text_orchestrator = BatchSubmissionOrchestrator(
            adapter=self._batch_adapter,
            config=replace(
                provider_cfg,
                metadata_output_path=self._with_metadata_suffix(
                    provider_cfg.metadata_output_path, "text", run_id
                ),
            ),
        )
        self._multimodal_orchestrator = BatchSubmissionOrchestrator(
            adapter=self._batch_adapter,
            config=replace(
                provider_cfg,
                metadata_output_path=self._with_metadata_suffix(
                    provider_cfg.metadata_output_path, "multimodal", run_id
                ),
            ),
        )

    @staticmethod
    def _with_metadata_suffix(path: str, suffix: str, run_id: str) -> str:
        if not path:
            return ""
        base_path = path.removesuffix(".jsonl")
        return f"{base_path}.{suffix}.{run_id}.jsonl"

    @property
    def batch_mode_enabled(self) -> bool:
        return self._text_orchestrator is not None and self._multimodal_orchestrator is not None

    def _next_custom_id(self, output_name: str, modality: str) -> str:
        self._batch_request_counter += 1
        return f"{output_name}:{modality}:{self._batch_request_counter}"
        # Cumulative token counts across all generate() calls in this processor's lifetime.
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0

    def get_load_time(self) -> float:
        """Return the wall-clock seconds spent initializing the SGLang engine."""
        return self._model_load_seconds

    def get_token_counts(self) -> TokenCounts:
        """Return cumulative token counts for this processor.

        Returns:
            TokenCounts object containing input and output token counts accumulated since this processor was created.
        """
        return TokenCounts(
            input_tokens=self._total_input_tokens,
            output_tokens=self._total_output_tokens
        )

    def _accumulate_tokens(self, outputs: list) -> None:
        """Add token counts from a list of SGLang generate() outputs."""
        for out in outputs:
            meta = out.get("meta_info") or {}
            self._total_input_tokens += int(meta.get("prompt_tokens") or 0)
            self._total_output_tokens += int(meta.get("completion_tokens") or 0)

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
        prompts_for_output: List[str] = []
        jinja_template = jinja2.Template(prompt_template)

        for var in vars_samples:
            user_prompt = [{"role": "user", "content": jinja_template.render(**var.to_dict())}]
            formatted = self.tokenizer.apply_chat_template(
                user_prompt, tokenize=False, add_generation_prompt=True
            )
            prompts_for_output.append(formatted)

        return prompts_for_output

    def build_multimodal_prompt(
        self, prompt_template: str, var_env: VariableEnvironment
    ) -> Tuple[str, Any]:
        """Build a prompt and extract images for SGLang Engine.

        Returns:
            (formatted_prompt, image_data_element)
        """
        jinja_template = jinja2.Template(prompt_template)
        base_prompt = jinja_template.render(**var_env.to_dict())

        # The image_data element must be aligned 1:1 with prompts.
        imgs = var_env.get_images()
        if not imgs:
            image_data_elem: Any = None
        elif len(imgs) == 1:
            image_data_elem = imgs[0]
        else:
            image_data_elem = imgs

        return base_prompt, image_data_elem

    def _get_image_token(self) -> str:
        """Get the image token for the current chat template."""
        if not self.chat_template:
            return "<image>"

        # Import chat templates from sglang if available
        try:
            from sglang.srt.conversation import chat_templates  # type: ignore

            if self.chat_template in chat_templates:
                conv = chat_templates[self.chat_template].copy()
                return conv.image_token
        except Exception:
            pass

        return IMAGE_TOKENS.get(self.chat_template, "<image>")

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
        nb_samples = len(batch)

        if self.batch_mode_enabled:
            return self._batch_process_sample(batch=batch, output_var=output_var)

        # Prepare sampling params
        sampling_params_output = self.sampling_params.copy()

        if output_var.output_type == "JSON":
            json_schema = output_var.get_output_schema()
            if json_schema is None:
                raise ValueError(
                    f"Output variable {output_var.name} has output_type=JSON but no output_schema defined."
                )
            sampling_params_output["json_schema"] = json.dumps(
                json_schema.model_json_schema()
            )

        # Separate samples into text-only and multimodal groups
        text_only_indices: List[int] = []
        multimodal_indices: List[int] = []
        for i in range(nb_samples):
            if batch[i].has_images():
                multimodal_indices.append(i)
            else:
                text_only_indices.append(i)

        results: dict[int, VariableEnvironment] = {}

        # Text-only batch
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

                self._accumulate_tokens(text_only_outputs)

                for local_idx, global_i in enumerate(text_only_indices):
                    value = text_only_outputs[local_idx].get("text", "").strip()
                    if output_var.output_type == "JSON":
                        try:
                            value = json.loads(value)
                        except json.JSONDecodeError:
                            value = {}
                    results[global_i] = batch[global_i].with_variable(output_var.name, value)

            except Exception as e:
                logger.error(
                    f"Batch generation failed for text-only samples in output '{output_var.name}': {e}"
                )
                for global_i in text_only_indices:
                    empty_val = {} if output_var.output_type == "JSON" else ""
                    results[global_i] = batch[global_i].with_variable(output_var.name, empty_val)

        # Multimodal batch
        if multimodal_indices:
            image_token = self._get_image_token()
            jinja_template = jinja2.Template(output_var.prompt)

            multimodal_prompts: List[str] = []
            multimodal_image_data: List[Any] = []

            for global_i in multimodal_indices:
                var_env = batch[global_i]
                base_prompt = jinja_template.render(**var_env.to_dict())

                # Format prompt with chat template
                user_prompt = [{"role": "user", "content": base_prompt}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    user_prompt, tokenize=False, add_generation_prompt=True
                )

                # Append image token (common pattern for VL templates)
                formatted_prompt = f"{formatted_prompt}\n{image_token}\n"
                multimodal_prompts.append(formatted_prompt)

                # `image_data` must be aligned 1:1 with prompts.
                imgs = var_env.get_images()
                if not imgs:
                    multimodal_image_data.append(None)
                elif len(imgs) == 1:
                    multimodal_image_data.append(imgs[0])
                else:
                    multimodal_image_data.append(imgs)

            try:
                multimodal_outputs = self.llm.generate(
                    prompt=multimodal_prompts,
                    sampling_params=sampling_params_output,
                    image_data=multimodal_image_data,
                )

                if not isinstance(multimodal_outputs, list) or len(multimodal_outputs) != len(multimodal_indices):
                    raise RuntimeError(
                        f"Mismatch between multimodal prompts and outputs for '{output_var.name}': "
                        f"{len(multimodal_prompts)} vs "
                        f"{len(multimodal_outputs) if isinstance(multimodal_outputs, list) else 'non-list'}"
                    )

                self._accumulate_tokens(multimodal_outputs)

                for local_idx, global_i in enumerate(multimodal_indices):
                    value = multimodal_outputs[local_idx].get("text", "").strip()
                    if output_var.output_type == "JSON":
                        try:
                            value = json.loads(value)
                        except json.JSONDecodeError:
                            value = {}
                    results[global_i] = batch[global_i].with_variable(output_var.name, value)

            except Exception as e:
                logger.error(
                    f"Batch generation failed for multimodal samples in output '{output_var.name}': {e}"
                )
                for global_i in multimodal_indices:
                    empty_val = {} if output_var.output_type == "JSON" else ""
                    results[global_i] = batch[global_i].with_variable(output_var.name, empty_val)

        return [results[i] for i in range(nb_samples)]

    def _batch_process_sample(
        self,
        batch: List[VariableEnvironment],
        output_var: LLMOutputVar,
    ) -> List[VariableEnvironment]:
        assert self._batch_provider_config is not None
        assert self._batch_adapter is not None
        assert self._text_orchestrator is not None
        assert self._multimodal_orchestrator is not None

        nb_samples = len(batch)
        text_only_indices: List[int] = []
        multimodal_indices: List[int] = []
        index_to_custom_id: Dict[int, str] = {}
        for i in range(nb_samples):
            if batch[i].has_images():
                multimodal_indices.append(i)
            else:
                text_only_indices.append(i)

        if text_only_indices:
            jinja_template = jinja2.Template(output_var.prompt)
            requests: List[Dict[str, Any]] = []
            source_indices: List[int] = []
            for global_i in text_only_indices:
                base_prompt = jinja_template.render(**batch[global_i].to_dict())
                payload = {
                    "messages": [
                        {
                            "role": "user",
                            "content": base_prompt,
                        }
                    ]
                }
                if output_var.output_type == "JSON" and output_var.output_schema:
                    payload["expected_schema"] = list(output_var.output_schema)
                custom_id = self._next_custom_id(output_var.name, "text")
                index_to_custom_id[global_i] = custom_id
                request = self._batch_adapter.build_request(
                    custom_id=custom_id,
                    payload=payload,
                    config=self._batch_provider_config,
                )
                requests.append(request)
                source_indices.append(self._global_row_offset + global_i)

            self._text_orchestrator.add_requests(
                requests=requests,
                source_indices=source_indices,
                model_params_snapshot={
                    "output_name": output_var.name,
                    "output_type": output_var.output_type,
                    "modality": "text",
                },
            )

        if multimodal_indices:
            requests = []
            source_indices = []
            for global_i in multimodal_indices:
                base_prompt, image_data = self.build_multimodal_prompt(output_var.prompt, batch[global_i])
                content: List[Dict[str, Any]] = [{"type": "text", "text": base_prompt}]

                if image_data is not None:
                    if isinstance(image_data, list):
                        images = image_data
                    else:
                        images = [image_data]
                    for image_ref in images:
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": str(image_ref)},
                            }
                        )

                payload = {
                    "messages": [
                        {
                            "role": "user",
                            "content": content,
                        }
                    ]
                }
                if output_var.output_type == "JSON" and output_var.output_schema:
                    payload["expected_schema"] = list(output_var.output_schema)

                custom_id = self._next_custom_id(output_var.name, "multimodal")
                index_to_custom_id[global_i] = custom_id
                request = self._batch_adapter.build_request(
                    custom_id=custom_id,
                    payload=payload,
                    config=self._batch_provider_config,
                )
                requests.append(dict(request))
                source_indices.append(self._global_row_offset + global_i)

            self._multimodal_orchestrator.add_requests(
                requests=requests,
                source_indices=source_indices,
                model_params_snapshot={
                    "output_name": output_var.name,
                    "output_type": output_var.output_type,
                    "modality": "multimodal",
                },
            )

        placeholders: List[VariableEnvironment] = []
        for i in range(nb_samples):
            unique_id = index_to_custom_id.get(i, f"unknown:{i}")
            placeholder = f"__BATCH_SUBMITTED__:{unique_id}"
            placeholders.append(batch[i].with_variable(output_var.name, placeholder))

        self._global_row_offset += nb_samples

        return placeholders

    def finalize(self) -> None:
        if not self.batch_mode_enabled:
            return

        assert self._text_orchestrator is not None
        assert self._multimodal_orchestrator is not None

        self._text_orchestrator.finalize(
            model_params_snapshot={
                "modality": "text",
                "phase": "finalize",
            }
        )
        self._multimodal_orchestrator.finalize(
            model_params_snapshot={
                "modality": "multimodal",
                "phase": "finalize",
            }
        )

    def shutdown(self) -> None:
        """Shutdown the LLM engine."""
        if self.llm is None:
            return

        try:
            self.llm.shutdown()
        except Exception as e:
            logger.warning(f"Error shutting down LLM: {e}")