"""Configuration for image generation processor in MMIRAGE."""

from dataclasses import dataclass, field
from enum import Enum

import logging
import os
from typing import Any, Dict, List, Optional, Sequence
from jinja2 import Environment, meta

from mmirage.core.process.base import BaseProcessorConfig
from mmirage.core.process.base import ProcessorRegistry
from mmirage.core.process.variables import BaseVar, OutputVar

logger = logging.getLogger(__name__)
env = Environment()


class ImageOutputMode(str, Enum):
    """Output representation for generated images."""

    PATH = "path"
    PIL = "pil"


@dataclass
class DiffusersPipelineArgs:
    """Runtime arguments used to initialize a Diffusers pipeline.

    Attributes:
        model_path: Hugging Face model id or local path.
        revision: Optional model revision (branch / tag / commit SHA).
        variant: Optional file variant, e.g. ``"fp16"`` for ``*.fp16.safetensors``.
        torch_dtype: Torch dtype as string. Common values: ``"float16"``,
            ``"bfloat16"``, ``"float32"``, ``"auto"``.
        device: Target device: ``"auto"``, ``"cuda"``, ``"cpu"``, or an
            explicit device string such as ``"cuda:1"``.
            ``"auto"`` distributes across all available GPUs when more than
            one is present (via ``device_map='auto'``), or falls back to CPU.
        enable_attention_slicing: Enable attention slicing to reduce VRAM
            usage.  Defaults to ``False`` because it can slow down modern
            CUDA setups; enable only when VRAM is constrained.
        local_files_only: If ``True``, only load from local cache; never
            contact the HuggingFace Hub.  Useful for air-gapped clusters.
        cache_dir: Override the HuggingFace cache directory.
        trust_remote_code: Allow custom model code from the Hub repository.
        custom_pipeline: Custom pipeline module name forwarded to
            ``from_pretrained``.
    """

    model_path: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    revision: Optional[str] = None
    variant: Optional[str] = None
    torch_dtype: str = "float16"
    device: str = "auto"
    enable_attention_slicing: bool = False
    local_files_only: bool = False
    cache_dir: Optional[str] = None
    trust_remote_code: bool = False
    custom_pipeline: Optional[str] = None


@dataclass
class SGLangBackendConfig:
    """Configuration for the SGLang Diffusion server backend.

    Two launch modes are supported:

    - ``external`` — MMIRAGE connects to an already-running SGLang server.
      The user is responsible for starting it before the pipeline runs.
    - ``managed`` — MMIRAGE spawns a local SGLang server as a subprocess,
      waits for it to become ready, and shuts it down when the shard finishes.
      This requires ``model_path`` and SGLang to be installed.

    Attributes:
        launch_mode: ``"external"`` or ``"managed"``.
        base_url: Base URL of the server (``http://host:port/v1``).  Ignored
            when ``launch_mode='managed'`` and ``port`` is set; inferred
            automatically in that case.
        api_key: ``Authorization: Bearer`` key.  ``"EMPTY"`` for local servers.
        timeout_seconds: Per-request HTTP timeout in seconds.
        model_path: HuggingFace model ID or local path forwarded to the server.
            Required for ``launch_mode='managed'``; optional for ``'external'``
            (sent as the ``model`` field in each request if supplied).
        port: Port the managed server should listen on.  Defaults to ``30010``.
        num_gpus: Tensor-parallelism degree (``--tp``).  Defaults to ``1``.
        dtype: Model weight dtype forwarded as ``--dtype``.  E.g. ``"float16"``.
        startup_timeout_seconds: Maximum seconds to wait for the managed server
            to become ready before raising an error.
        extra_server_args: Additional CLI arguments appended verbatim to the
            ``python -m sglang.launch_server`` command, e.g.
            ``["--mem-fraction-static", "0.9"]``.
    """

    launch_mode: str = "external"
    base_url: str = "http://127.0.0.1:30010/v1"
    api_key: str = "EMPTY"
    timeout_seconds: int = 900
    model_path: Optional[str] = None

    # managed-mode fields
    port: int = 30010
    num_gpus: int = 1
    dtype: Optional[str] = None
    startup_timeout_seconds: int = 120
    extra_server_args: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.launch_mode not in ("external", "managed"):
            raise ValueError(
                f"Unsupported SGLang launch_mode={self.launch_mode!r}. "
                "Choose 'external' (server already running) or "
                "'managed' (MMIRAGE starts the server automatically)."
            )
        if self.launch_mode == "managed" and not self.model_path:
            raise ValueError(
                "launch_mode='managed' requires model_path to be set so MMIRAGE "
                "knows which model to pass to the SGLang server."
            )
        if self.launch_mode == "managed":
            # Derive base_url from port so users don't have to repeat it.
            self.base_url = f"http://127.0.0.1:{self.port}/v1"


@dataclass
class ImageGenConfig(BaseProcessorConfig):
    """Configuration for the backend-neutral image generation processor.

    Attributes:
        backend: Image generation backend to use.  One of ``"diffusers"``
            (in-process Diffusers pipeline) or ``"sglang"`` (local SGLang
            Diffusion server).
        pipeline_args: Diffusers pipeline arguments (used when
            ``backend="diffusers"``).
        sglang: SGLang server configuration (used when ``backend="sglang"``).
        default_sampling_params: Default generation kwargs forwarded to every
            pipeline/server call (e.g. ``num_inference_steps``,
            ``guidance_scale``).
        parallel_inference: If ``True``, generate a full chunk of prompts in a
            single batched pipeline call (Diffusers backend) or concurrent
            server calls.  Chunks that fail are retried sample-by-sample.
        parallel_chunk_size: Maximum number of samples per batched call.
            ``None`` means use the full mapper batch size.
        output_dir: Directory where generated images are written when
            ``output_mode="path"``.  Supports ``~`` expansion.
        file_format: Image file format for saved outputs (e.g. ``"png"``,
            ``"jpg"``).
    """

    backend: str = "diffusers"
    pipeline_args: DiffusersPipelineArgs = field(default_factory=DiffusersPipelineArgs)
    sglang: Optional[SGLangBackendConfig] = None
    default_sampling_params: Dict[str, Any] = field(default_factory=dict)
    parallel_inference: bool = True
    parallel_chunk_size: Optional[int] = 4
    output_dir: str = "~/.cache/MMIRAGE/generated_images"
    file_format: str = "png"

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.backend not in ("diffusers", "sglang"):
            raise ValueError(
                f"Unsupported image_gen backend={self.backend!r}. "
                "Choose 'diffusers' or 'sglang'."
            )
        if self.backend == "sglang" and self.sglang is None:
            raise ValueError(
                "backend='sglang' requires a 'sglang:' configuration block."
            )
        if self.parallel_chunk_size is not None and self.parallel_chunk_size <= 0:
            raise ValueError(
                f"parallel_chunk_size must be a positive integer, got {self.parallel_chunk_size!r}. "
                "Set to None to use the full batch size."
            )
        if not self.file_format:
            logger.warning("file_format is empty; defaulting to 'png'.")
            self.file_format = "png"

    def get_output_dir(self) -> str:
        """Return the normalised absolute output directory path."""
        return os.path.abspath(os.path.expanduser(self.output_dir))


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------
#: ``DiffusersImageGenConfig`` is a legacy alias for :class:`ImageGenConfig`.
DiffusersImageGenConfig = ImageGenConfig


@dataclass
class ImageGenOutputVar(OutputVar):
    """Output variable generated by image generation processor.

    Attributes:
        prompt: Jinja2 template used as positive prompt.
        negative_prompt: Optional Jinja2 template used as negative prompt.
        output_mode: Output representation: "path" (default) or "pil".
        filename_template: Optional Jinja2 template used for saved image filename stem.
            Supported internal variables: ``__sample_index`` (shard-local row index,
            i.e. the position within this shard's output — combine with ``__shard_id``
            for global uniqueness), ``__output_name``, ``__shard_id``,
            ``__source_hash`` (8-char SHA-256 of input values).
            All input variables (e.g. ``text``) are also available.
        width: Optional image width override.
        height: Optional image height override.
        num_inference_steps: Optional sampling steps override.
        guidance_scale: Optional guidance scale override.
        seed: Optional deterministic seed. If set, sample index is added for uniqueness.
    """

    prompt: str = ""
    negative_prompt: str = ""
    output_mode: ImageOutputMode = ImageOutputMode.PATH
    filename_template: str = "generated_{{ __shard_id }}_{{ __sample_index }}_{{ __source_hash }}"
    width: Optional[int] = None
    height: Optional[int] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None

    def is_computable(self, vars: Sequence[BaseVar]) -> bool:
        """Check if all variables referenced in templates are available."""
        reserved = {"__sample_index", "__output_name", "__shard_id", "__source_hash"}
        var_names = {v.name for v in vars}

        # Prompt/negative_prompt are rendered from env.to_dict() only — reserved
        # vars are not injected there, so treat them as undeclared in those templates.
        prompt_templates: List[str] = [self.prompt]
        if self.negative_prompt:
            prompt_templates.append(self.negative_prompt)

        undeclared: set[str] = set()
        for template in prompt_templates:
            parsed_content = env.parse(template)
            template_vars = meta.find_undeclared_variables(parsed_content)
            undeclared |= template_vars - var_names

        # filename_template is rendered with reserved vars injected, so allow them.
        if self.filename_template:
            parsed_content = env.parse(self.filename_template)
            template_vars = meta.find_undeclared_variables(parsed_content)
            undeclared |= template_vars - var_names - reserved

        if undeclared:
            logger.warning(
                f"⚠️ Undeclared variables found for {self.name}: {undeclared}"
            )
            return False

        if self.output_mode not in set(ImageOutputMode):
            logger.warning(
                f"⚠️ Invalid output_mode for {self.name}: {self.output_mode}. Expected one of {[m.value for m in ImageOutputMode]}"
            )
            return False

        return True


ProcessorRegistry.register_types("image_gen", ImageGenConfig, ImageGenOutputVar)
