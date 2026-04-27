"""Image generation processor implementation using Diffusers."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import socket
import tempfile
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

    def __init__(self, config: DiffusersImageGenConfig, shard_id: int = 0, **kwargs) -> None:
        """Initialize processor and load Diffusers pipeline."""
        super().__init__(config)

        try:
            import torch
            from diffusers import DiffusionPipeline  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "image_gen processor requires optional dependencies. "
                "Install with: pip install -e .[image_gen]"
            ) from e

        self._torch = torch
        self._pipeline, self._device = self._build_pipeline(DiffusionPipeline, config)
        self._default_sampling_params = dict(config.default_sampling_params)
        self._parallel_inference = bool(config.parallel_inference)
        self._parallel_chunk_size = config.parallel_chunk_size

        self._output_dir = config.get_output_dir()
        self._file_format = (config.file_format or "png").lower()
        os.makedirs(self._output_dir, exist_ok=True)

        self._shard_id = shard_id
        self._sample_counter = 0
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

        return pipeline, device

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

    def _build_batch_sampling_params(
        self, output_var: ImageGenOutputVar, start_index: int, chunk_size: int
    ) -> Dict[str, Any]:
        """Build generation kwargs for a batched inference call."""
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
            generators = []
            for offset in range(chunk_size):
                generator = self._torch.Generator(device=self._device)
                generator = generator.manual_seed(int(output_var.seed) + start_index + offset)
                generators.append(generator)
            params["generator"] = generators

        return params

    @staticmethod
    def _compute_source_hash(env: VariableEnvironment) -> str:
        """Return an 8-character SHA-256 hex digest of all input variable values."""
        payload = str(sorted(env.to_dict().items()))
        return hashlib.sha256(payload.encode()).hexdigest()[:8]

    def _render_filename(self, filename_template: jinja2.Template, output_var: ImageGenOutputVar, env: VariableEnvironment, sample_index: int) -> str:
        """Render output filename stem from template and context."""
        context = dict(env.to_dict())
        context["__sample_index"] = sample_index
        context["__output_name"] = output_var.name
        context["__shard_id"] = self._shard_id
        context["__source_hash"] = self._compute_source_hash(env)

        stem = filename_template.render(**context)
        stem = _sanitize_filename(stem)
        return f"{stem}.{self._file_format}"

    def _save_image(self, image: Any, filename: str) -> str:
        """Persist image to output directory and return absolute path."""
        stem, ext = os.path.splitext(filename)
        path = os.path.join(self._output_dir, filename)
        if os.path.exists(path):
            path = os.path.join(self._output_dir, f"{stem}.{self._run_id}{ext}")

        tmp_fd, tmp_path = tempfile.mkstemp(dir=self._output_dir, suffix=ext)
        try:
            os.close(tmp_fd)
            image.save(tmp_path)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        return path

    def _process_chunk_parallel(
        self,
        chunk: List[VariableEnvironment],
        output_var: ImageGenOutputVar,
        prompt_template: jinja2.Template,
        negative_prompt_template: Optional[jinja2.Template],
        start_index: int,
        filename_template: jinja2.Template,
    ) -> List[VariableEnvironment]:
        """Process a chunk of samples with a single batched Diffusers call."""
        prompts: List[str] = []
        negative_prompts: Optional[List[str]] = [] if negative_prompt_template is not None else None

        for env in chunk:
            context = env.to_dict()
            prompts.append(prompt_template.render(**context))
            if negative_prompt_template is not None and negative_prompts is not None:
                negative_prompts.append(negative_prompt_template.render(**context))

        sampling_params = self._build_batch_sampling_params(output_var, start_index, len(chunk))
        call_kwargs: Dict[str, Any] = {"prompt": prompts, **sampling_params}
        if negative_prompts is not None:
            call_kwargs["negative_prompt"] = negative_prompts
        output = self._pipeline(**call_kwargs)

        images = output.images
        if len(images) != len(chunk):
            raise RuntimeError(
                f"Expected {len(chunk)} images from batched generation, got {len(images)}"
            )

        updated: List[VariableEnvironment] = []
        for local_index, (env, image) in enumerate(zip(chunk, images)):
            sample_index = start_index + local_index
            if output_var.output_mode == "pil":
                value = image
            else:
                filename = self._render_filename(filename_template, output_var, env, sample_index)
                value = self._save_image(image, filename)

            updated.append(env.with_variable(output_var.name, value, is_image=True))

        return updated

    def _batch_process_parallel(
        self,
        batch: List[VariableEnvironment],
        output_var: ImageGenOutputVar,
        prompt_template: jinja2.Template,
        negative_prompt_template: Optional[jinja2.Template],
        filename_template: jinja2.Template,
    ) -> List[VariableEnvironment]:
        """Process a full mapper batch using batched Diffusers calls."""
        chunk_size = self._parallel_chunk_size or len(batch)
        updated: List[VariableEnvironment] = []

        for start_index in range(0, len(batch), chunk_size):
            chunk = batch[start_index : start_index + chunk_size]
            updated.extend(
                self._process_chunk_parallel(
                    chunk,
                    output_var,
                    prompt_template,
                    negative_prompt_template,
                    self._sample_counter + start_index,
                    filename_template,
                )
            )

        self._sample_counter += len(batch)
        return updated

    def _batch_process_sequential(
        self,
        batch: List[VariableEnvironment],
        output_var: ImageGenOutputVar,
        prompt_template: jinja2.Template,
        negative_prompt_template: Optional[jinja2.Template],
        filename_template: jinja2.Template,
    ) -> List[VariableEnvironment]:
        """Process samples sequentially (legacy behavior)."""
        updated: List[VariableEnvironment] = []

        for local_index, env in enumerate(batch):
            sample_index = self._sample_counter + local_index
            try:
                context = env.to_dict()
                prompt = prompt_template.render(**context)
                negative_prompt = (
                    negative_prompt_template.render(**context)
                    if negative_prompt_template is not None
                    else None
                )

                sampling_params = self._build_sampling_params(output_var, sample_index)
                call_kwargs: Dict[str, Any] = {"prompt": prompt, **sampling_params}
                if negative_prompt is not None:
                    call_kwargs["negative_prompt"] = negative_prompt
                output = self._pipeline(**call_kwargs)
                image = output.images[0]

                if output_var.output_mode == "pil":
                    value = image
                else:
                    filename = self._render_filename(filename_template, output_var, env, sample_index)
                    value = self._save_image(image, filename)

                updated.append(env.with_variable(output_var.name, value, is_image=True))
            except Exception as exc:
                logger.error(
                    f"Image generation failed for output '{output_var.name}' at sample {sample_index}: {exc}"
                )
                updated.append(
                    env.with_variable(output_var.name, None, is_image=True)
                )

        self._sample_counter += len(batch)
        return updated

    @override
    def batch_process_sample(
        self, batch: List[VariableEnvironment], output_var: ImageGenOutputVar
    ) -> List[VariableEnvironment]:
        """Generate images for each sample in the batch."""
        prompt_template = jinja2.Template(output_var.prompt)
        negative_prompt_template = (
            jinja2.Template(output_var.negative_prompt)
            if output_var.negative_prompt
            else None
        )
        filename_template = jinja2.Template(output_var.filename_template)

        if self._parallel_inference and len(batch) > 1:
            try:
                return self._batch_process_parallel(
                    batch,
                    output_var,
                    prompt_template,
                    negative_prompt_template,
                    filename_template,
                )
            except Exception as exc:
                logger.warning(
                    "Parallel image generation failed; falling back to sequential mode. "
                    f"Reason: {exc}"
                )

        return self._batch_process_sequential(
            batch,
            output_var,
            prompt_template,
            negative_prompt_template,
            filename_template,
        )

    def shutdown(self) -> None:
        """Release pipeline references."""
        self._pipeline = None
