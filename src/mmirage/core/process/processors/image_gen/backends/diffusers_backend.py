"""Diffusers-based image generation backend."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from mmirage.core.process.processors.image_gen.config import DiffusersPipelineArgs

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from diffusers import DiffusionPipeline

logger = logging.getLogger(__name__)


class DiffusersImageBackend:
    """Image generation backend using an in-process Diffusers pipeline.

    This backend loads a local or cached Diffusers pipeline once at
    construction time and keeps it in memory for repeated batched generation.
    """

    def __init__(self, pipeline_args: DiffusersPipelineArgs) -> None:
        try:
            import torch
            from diffusers import DiffusionPipeline  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "diffusers backend requires optional dependencies. "
                "Install with: pip install -e .[image_gen]"
            ) from exc

        self._torch = torch
        self._pipeline, self._generator_device = self._build_pipeline(
            DiffusionPipeline,
            pipeline_args,
        )

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------

    def _build_pipeline(
        self,
        pipeline_cls: type[DiffusionPipeline],
        args: DiffusersPipelineArgs,
    ) -> Tuple[DiffusionPipeline, str]:
        """Load and configure the Diffusers pipeline.

        Returns:
            ``(pipeline, generator_device)`` where ``generator_device`` is the
            device string used for ``torch.Generator`` objects.

        Notes:
            ``device_map`` is only local-process model/component placement. It
            is not Slurm multi-node distribution.
        """
        load_kwargs = self._build_load_kwargs(args)

        placement_device = args.device
        generator_device = args.device
        use_device_map = False

        if placement_device == "auto":
            placement_device, generator_device, use_device_map = self._resolve_auto_device(args)

            if use_device_map:
                device_map = getattr(args, "device_map", None) or "balanced"
                load_kwargs["device_map"] = device_map

                logger.info(
                    "device='auto': using Diffusers device_map=%r across %d visible GPUs",
                    device_map,
                    self._torch.cuda.device_count(),
                )

        pipeline = pipeline_cls.from_pretrained(args.model_path, **load_kwargs)

        if not use_device_map:
            pipeline = pipeline.to(placement_device)

        if getattr(args, "enable_attention_slicing", False):
            if hasattr(pipeline, "enable_attention_slicing"):
                pipeline.enable_attention_slicing()
                logger.info("Enabled Diffusers attention slicing.")
            else:
                logger.warning(
                    "enable_attention_slicing=True was requested, but this pipeline "
                    "does not expose enable_attention_slicing()."
                )

        return pipeline, generator_device

    def _build_load_kwargs(self, args: DiffusersPipelineArgs) -> Dict[str, Any]:
        """Build keyword arguments for ``DiffusionPipeline.from_pretrained``."""
        load_kwargs: Dict[str, Any] = {}

        torch_dtype = self._parse_torch_dtype(getattr(args, "torch_dtype", "auto"))
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype

        optional_fields = {
            "revision": "revision",
            "cache_dir": "cache_dir",
            "custom_pipeline": "custom_pipeline",
            "variant": "variant",
        }
        for attr_name, kwarg_name in optional_fields.items():
            value = getattr(args, attr_name, None)
            if value:
                load_kwargs[kwarg_name] = value

        boolean_fields = {
            "local_files_only": "local_files_only",
            "trust_remote_code": "trust_remote_code",
        }
        for attr_name, kwarg_name in boolean_fields.items():
            value = getattr(args, attr_name, False)
            if value:
                load_kwargs[kwarg_name] = True

        return load_kwargs

    def _resolve_auto_device(self, args: DiffusersPipelineArgs) -> Tuple[str, str, bool]:
        """Resolve ``device='auto'`` into placement/generator choices.

        Returns:
            ``(placement_device, generator_device, use_device_map)``.
        """
        if not self._torch.cuda.is_available():
            logger.info("device='auto': resolved to 'cpu' because CUDA is unavailable.")
            return "cpu", "cpu", False

        num_gpus = self._torch.cuda.device_count()
        if num_gpus <= 0:
            logger.info("device='auto': resolved to 'cpu' because no CUDA devices are visible.")
            return "cpu", "cpu", False

        if num_gpus == 1:
            logger.info("device='auto': resolved to 'cuda' because one GPU is visible.")
            return "cuda", "cuda", False

        # Multiple GPUs visible to this process. This is local model/component
        # placement, not distributed Slurm execution.
        #
        # CPU generators are safest when the model is split across multiple
        # devices and also improve reproducibility across CPU/GPU setups.
        return "cpu", "cpu", True

    def _parse_torch_dtype(self, dtype: str) -> Optional[torch.dtype]:
        """Convert a dtype string into a ``torch.dtype``."""
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
                "Use one of: auto, float16, bfloat16, float32."
            )

        return mapping[dtype_key]

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    def _build_generators(self, seeds: List[Optional[int]]) -> Optional[Any]:
        """Build one or more ``torch.Generator`` objects from seeds.

        Returns:
            ``None`` when all seeds are ``None``.
            A single generator for one prompt.
            A list of generators for batched prompts.
        """
        if not seeds or all(seed is None for seed in seeds):
            return None

        if any(seed is None for seed in seeds):
            raise ValueError(
                "Diffusers batched generation requires seeds to be either all set "
                "or all None. Received a mixed seed list."
            )

        generators = []
        for seed in seeds:
            generator = self._torch.Generator(device=self._generator_device)
            generator.manual_seed(int(seed))
            generators.append(generator)

        return generators[0] if len(generators) == 1 else generators

    @staticmethod
    def _normalize_negative_prompts(
        prompts: List[str],
        negative_prompts: Optional[List[Optional[str]]],
    ) -> Optional[List[str]]:
        """Normalize optional negative prompts for Diffusers."""
        if negative_prompts is None:
            return None

        if len(negative_prompts) != len(prompts):
            raise ValueError(
                f"Expected {len(prompts)} negative prompts, got {len(negative_prompts)}"
            )

        return [negative_prompt or "" for negative_prompt in negative_prompts]

    # ------------------------------------------------------------------
    # Backend interface
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        prompts: List[str],
        negative_prompts: Optional[List[Optional[str]]],
        params: Dict[str, Any],
        seeds: List[Optional[int]],
    ) -> List[Any]:
        """Generate images via the in-process Diffusers pipeline.

        The pipeline is invoked once with a list of prompts, allowing Diffusers
        to perform batched generation on the selected local device placement.
        """
        if not prompts:
            return []

        if seeds and len(seeds) != len(prompts):
            raise ValueError(f"Expected {len(prompts)} seeds, got {len(seeds)}")

        call_kwargs: Dict[str, Any] = dict(params)
        call_kwargs["prompt"] = prompts

        normalized_negative_prompts = self._normalize_negative_prompts(
            prompts,
            negative_prompts,
        )
        if normalized_negative_prompts is not None:
            call_kwargs["negative_prompt"] = normalized_negative_prompts

        generators = self._build_generators(seeds)
        if generators is not None:
            call_kwargs["generator"] = generators

        output = self._pipeline(**call_kwargs)
        images = output.images

        if len(images) != len(prompts):
            raise RuntimeError(
                f"Expected {len(prompts)} images from Diffusers pipeline, got {len(images)}"
            )

        return images

    def shutdown(self) -> None:
        """Release the pipeline reference and free CUDA cache."""
        self._pipeline = None

        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
            logger.info("Released Diffusers pipeline and emptied CUDA cache.")