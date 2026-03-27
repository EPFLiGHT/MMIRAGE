"""Image generation processor implementation using Diffusers."""

from __future__ import annotations

import logging
import os
import re
import socket
import uuid
from typing import Any, Dict, List, Optional

import jinja2

from mmirage.core.process.base import BaseProcessor, ProcessorRegistry
from mmirage.core.process.processors.image_gen.config import (
    DiffusersImageGenConfig,
    ImageGenOutputVar,
)
from mmirage.core.process.variables import VariableEnvironment

try:
    from typing import override  # Python 3.12+
except ImportError:  # pragma: no cover
    from typing_extensions import override  # type: ignore


logger = logging.getLogger(__name__)

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_filename(filename: str) -> str:
    """Return a filesystem-safe filename stem."""
    normalized = _SAFE_FILENAME_RE.sub("_", filename).strip("._")
    return normalized or "image"


@ProcessorRegistry.register("image_gen", DiffusersImageGenConfig, ImageGenOutputVar)
class ImageGenProcessor(BaseProcessor[ImageGenOutputVar]):
    """Processor that generates images from prompts using Diffusers.

    This processor currently supports text-to-image generation and returns
    either saved file paths or in-memory PIL.Image outputs.
    """

    def __init__(self, config: DiffusersImageGenConfig, **kwargs) -> None:
        """Initialize processor and load Diffusers pipeline."""
        super().__init__(config, **kwargs)

        try:
            import torch
            from diffusers import DiffusionPipeline  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "image_gen processor requires optional dependencies. "
                "Install with: pip install diffusers accelerate safetensors"
            ) from e

        self._torch = torch
        self._pipeline = self._build_pipeline(DiffusionPipeline, config)
        self._default_sampling_params = dict(config.default_sampling_params)

        self._output_dir = config.get_output_dir()
        self._file_format = (config.file_format or "png").lower()
        os.makedirs(self._output_dir, exist_ok=True)

        run_token = uuid.uuid4().hex[:8]
        self._run_id = f"{socket.gethostname()}.{os.getpid()}.{run_token}"

    def _build_pipeline(self, pipeline_cls: Any, config: DiffusersImageGenConfig) -> Any:
        """Construct and move Diffusers pipeline to target device."""
        pipeline_args = config.pipeline_args
        load_kwargs: Dict[str, Any] = {}

        torch_dtype = self._parse_torch_dtype(pipeline_args.torch_dtype)
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype

        if pipeline_args.revision:
            load_kwargs["revision"] = pipeline_args.revision

        pipeline = pipeline_cls.from_pretrained(pipeline_args.model_path, **load_kwargs)

        device = pipeline_args.device
        if device == "auto":
            device = "cuda" if self._torch.cuda.is_available() else "cpu"

        pipeline = pipeline.to(device)

        if pipeline_args.enable_attention_slicing and hasattr(pipeline, "enable_attention_slicing"):
            pipeline.enable_attention_slicing()

        self._device = device
        return pipeline

    def _parse_torch_dtype(self, dtype: str) -> Optional[Any]:
        """Convert dtype string to torch dtype object."""
        if not dtype:
            return None

        dtype_key = dtype.lower()
        if dtype_key == "auto":
            return None

        mapping = {
            "float16": self._torch.float16,
            "fp16": self._torch.float16,
            "bfloat16": self._torch.bfloat16,
            "bf16": self._torch.bfloat16,
            "float32": self._torch.float32,
            "fp32": self._torch.float32,
        }

        if dtype_key not in mapping:
            raise ValueError(
                f"Unsupported torch_dtype={dtype!r}. "
                "Use one of: auto, float16, bfloat16, float32"
            )

        return mapping[dtype_key]

    def _build_sampling_params(self, output_var: ImageGenOutputVar, sample_index: int) -> Dict[str, Any]:
        """Build generation kwargs for a single sample."""
        params = dict(self._default_sampling_params)

        if output_var.width is not None:
            params["width"] = output_var.width
        if output_var.height is not None:
            params["height"] = output_var.height
        if output_var.num_inference_steps is not None:
            params["num_inference_steps"] = output_var.num_inference_steps
        if output_var.guidance_scale is not None:
            params["guidance_scale"] = output_var.guidance_scale

        if output_var.seed is not None:
            generator = self._torch.Generator(device=self._device)
            generator = generator.manual_seed(int(output_var.seed) + sample_index)
            params["generator"] = generator

        return params

    def _render_filename(self, output_var: ImageGenOutputVar, env: VariableEnvironment, sample_index: int) -> str:
        """Render output filename stem from template and context."""
        template = jinja2.Template(output_var.filename_template)
        context = dict(env.to_dict())
        context["__sample_index"] = sample_index
        context["__output_name"] = output_var.name

        stem = template.render(**context)
        stem = _sanitize_filename(stem)
        return f"{stem}.{self._file_format}"

    def _save_image(self, image: Any, filename: str) -> str:
        """Persist image to output directory and return absolute path."""
        path = os.path.join(self._output_dir, filename)

        if os.path.exists(path):
            stem, ext = os.path.splitext(filename)
            path = os.path.join(self._output_dir, f"{stem}.{self._run_id}{ext}")

        image.save(path)
        return path

    @override
    def batch_process_sample(
        self, batch: List[VariableEnvironment], output_var: ImageGenOutputVar
    ) -> List[VariableEnvironment]:
        """Generate images for each sample in the batch."""
        updated: List[VariableEnvironment] = []
        prompt_template = jinja2.Template(output_var.prompt)
        negative_prompt_template = (
            jinja2.Template(output_var.negative_prompt)
            if output_var.negative_prompt
            else None
        )

        for sample_index, env in enumerate(batch):
            try:
                context = env.to_dict()
                prompt = prompt_template.render(**context)
                negative_prompt = (
                    negative_prompt_template.render(**context)
                    if negative_prompt_template is not None
                    else None
                )

                sampling_params = self._build_sampling_params(output_var, sample_index)
                output = self._pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    **sampling_params,
                )
                image = output.images[0]

                if output_var.output_mode == "pil":
                    value = image
                    is_image = True
                else:
                    filename = self._render_filename(output_var, env, sample_index)
                    value = self._save_image(image, filename)
                    is_image = False

                updated.append(env.with_variable(output_var.name, value, is_image=is_image))
            except Exception as exc:
                logger.error(
                    f"Image generation failed for output '{output_var.name}' at sample {sample_index}: {exc}"
                )
                fallback_value = None if output_var.output_mode == "pil" else ""
                updated.append(
                    env.with_variable(output_var.name, fallback_value, is_image=False)
                )

        return updated

    def shutdown(self) -> None:
        """Release pipeline references."""
        self._pipeline = None
