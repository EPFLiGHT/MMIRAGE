"""API LLM processor implementation using OpenAI-compatible endpoints."""

from __future__ import annotations

import base64
import io
import json
import logging
import mimetypes
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import jinja2
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from PIL import Image

from mmirage.core.process.base import BaseProcessor, ProcessorRegistry
from mmirage.core.process.processors.api_llm.config import APILLMConfig, APILLMOutputVar
from mmirage.core.process.variables import VariableEnvironment

try:
    from typing import override
except ImportError:
    from typing_extensions import override

logger = logging.getLogger(__name__)

_RETRYABLE_EXCEPTIONS = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)


def _image_to_base64_data_url(image: Any) -> str:
    if isinstance(image, str):
        if image.startswith(("http://", "https://")):
            return image
        mime_type, _ = mimetypes.guess_type(image)
        mime_type = mime_type or "image/png"
        with open(image, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"
    elif isinstance(image, Image.Image):
        buffered = io.BytesIO()
        fmt = image.format or "PNG"
        image.save(buffered, format=fmt)
        mime_type = f"image/{fmt.lower()}"
        encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")


@ProcessorRegistry.register("api_llm", APILLMConfig, APILLMOutputVar)
class APILLMProcessor(BaseProcessor[APILLMOutputVar]):
    """API-based LLM processor using the OpenAI Python client."""

    def __init__(self, config: APILLMConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        api_key = os.environ.get(config.api_key_env, "")
        if not api_key:
            logger.warning(
                f"API key environment variable '{config.api_key_env}' is not set or empty."
            )
        client_kwargs: Dict[str, Any] = {"api_key": api_key or "none"}
        if config.base_url:
            client_kwargs["base_url"] = config.base_url
        self.client = OpenAI(**client_kwargs)
        self.model = config.model
        self.sampling_params = config.default_sampling_params
        self.max_concurrency = config.max_concurrency
        self.max_retries = config.max_retries
        self.retry_base_delay = config.retry_base_delay

    def _build_messages(
        self, prompt_template: str, var_env: VariableEnvironment
    ) -> List[Dict[str, Any]]:
        jinja_template = jinja2.Template(prompt_template)
        text_content = jinja_template.render(**var_env.to_dict())

        if not var_env.has_images():
            return [{"role": "user", "content": text_content}]

        content_parts: List[Dict[str, Any]] = [{"type": "text", "text": text_content}]
        for img in var_env.get_images():
            url = _image_to_base64_data_url(img)
            content_parts.append({"type": "image_url", "image_url": {"url": url}})

        return [{"role": "user", "content": content_parts}]

    def _call_api_with_retry(
        self,
        messages: List[Dict[str, Any]],
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            **self.sampling_params,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        last_exception: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
            except _RETRYABLE_EXCEPTIONS as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = self.retry_base_delay * (2 ** attempt)
                    logger.warning(
                        f"API call failed (attempt {attempt + 1}/{self.max_retries + 1}): "
                        f"{type(e).__name__}: {e}. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"API call failed after {self.max_retries + 1} attempts: "
                        f"{type(e).__name__}: {e}"
                    )
        raise last_exception  # type: ignore[misc]

    @override
    def batch_process_sample(
        self, batch: List[VariableEnvironment], output_var: APILLMOutputVar
    ) -> List[VariableEnvironment]:
        nb_samples = len(batch)

        response_format: Optional[Dict[str, Any]] = None
        if output_var.output_type == "JSON":
            json_schema = output_var.get_output_schema()
            if json_schema is None:
                raise ValueError(
                    f"Output variable {output_var.name} has output_type=JSON "
                    f"but no output_schema defined."
                )
            schema = json_schema.model_json_schema()
            schema["additionalProperties"] = False
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "output_schema",
                    "schema": schema,
                    "strict": True,
                },
            }

        all_messages = [
            self._build_messages(output_var.prompt, var_env) for var_env in batch
        ]

        results: Dict[int, str] = {}

        def _process_one(idx: int) -> tuple[int, str]:
            text = self._call_api_with_retry(all_messages[idx], response_format)
            return idx, text

        with ThreadPoolExecutor(max_workers=min(self.max_concurrency, nb_samples)) as pool:
            futures = {pool.submit(_process_one, i): i for i in range(nb_samples)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    _, text = future.result()
                    results[idx] = text
                except Exception as e:
                    logger.error(
                        f"API call failed for sample {idx} in output '{output_var.name}': {e}"
                    )

        output_envs: List[VariableEnvironment] = []
        for i in range(nb_samples):
            if i in results:
                value: Any = results[i].strip()
                if output_var.output_type == "JSON":
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        value = {}
            else:
                value = {} if output_var.output_type == "JSON" else ""
            output_envs.append(batch[i].with_variable(output_var.name, value))

        return output_envs

    def shutdown(self) -> None:
        pass
