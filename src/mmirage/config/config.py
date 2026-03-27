"""Configuration dataclasses for MMIRAGE pipeline."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mmirage.config.loading import LoadingParams
from mmirage.core.process.base import BaseProcessorConfig
from mmirage.core.process.variables import InputVar, OutputVar


@dataclass
class ExecutionParams:
    """Parameters for executing the MMIRAGE pipeline.

    Defines how the pipeline is executed, including local or SLURM-based
    distributed execution, retry logic, and resource allocation.

    Attributes:
        mode: Execution mode: "local" or "slurm". Defaults to "local".
        retry: Whether automatic retry orchestration is enabled. Defaults to False.
        max_retries: Maximum number of retries for failed shards. Defaults to 3.
        poll_interval_seconds: Seconds to wait between polling job status. Defaults to 30.
        settle_time_seconds: Seconds to wait after job completes before checking results. Defaults to 60.

        # SLURM-specific parameters
        account: HPC account/partition to charge. Required for SLURM mode.
        job_name: SLURM job name. Defaults to "mmirage-sharded".
        reservation: Optional SLURM reservation name.
        nodes: Number of nodes. Defaults to 1.
        ntasks_per_node: Number of tasks per node. Defaults to 1.
        gpus: Number of GPUs per node. Defaults to 4.
        cpus_per_task: Number of CPUs per task. Defaults to 288.
        time_limit: Job time limit (HH:MM:SS). Defaults to "11:59:59".

        # Paths
        project_root: Base project directory. Can use environment variables with ${VAR}.
        report_dir: Directory for SLURM output/error files. Defaults to ~/reports.
        hf_home: HuggingFace cache directory. Defaults to ~/hf.
        edf_env: Optional EDF environment file path.
    """

    mode: str = "local"
    retry: bool = False
    max_retries: int = 3
    poll_interval_seconds: int = 30
    settle_time_seconds: int = 60

    # Paths (can contain environment variables like ${VAR} or $VAR)
    project_root: Optional[str] = None
    report_dir: str = "~/reports"
    hf_home: str = "~/hf"
    edf_env: Optional[str] = None

    # SLURM parameters
    account: Optional[str] = None
    job_name: str = "mmirage-sharded"
    reservation: Optional[str] = None
    nodes: int = 1
    ntasks_per_node: int = 1
    gpus: int = 4
    cpus_per_task: int = 288
    time_limit: str = "11:59:59"

    def __post_init__(self):
        """Validate execution parameters."""
        if self.mode not in ("local", "slurm"):
            raise ValueError(f"Invalid execution mode: {self.mode!r}. Must be 'local' or 'slurm'.")
        if self.mode == "slurm" and not self.account:
            raise ValueError("account is required when mode='slurm'")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")

    def is_slurm(self) -> bool:
        """Check if execution mode is SLURM."""
        return self.mode == "slurm"


@dataclass
class ProcessingParams:
    """Parameters for processing dataset samples.

    Defines how input variables are extracted, outputs are generated,
    and the final output schema is constructed.

    Attributes:
        inputs: List of input variables to extract from source datasets.
        outputs: List of output variables to generate using processors.
        output_schema: Dictionary defining the structure of output samples.
        remove_columns: If True, removes all columns from original dataset.
    """

    inputs: List[InputVar]
    outputs: List[OutputVar]
    output_schema: Dict[str, Any]
    remove_columns: bool = False


@dataclass
class ImageCanonicalConfig:
    """Canonicalization settings for image assets."""

    apply_exif_orientation: bool = True
    convert_mode: Optional[str] = "RGB"
    max_side: Optional[int] = None
    format: str = "png"
    jpeg_quality: int = 95

    def __post_init__(self) -> None:
        self.format = self.format.lower().strip()
        if self.format == "jpg":
            self.format = "jpeg"
        if self.format not in ("png", "jpeg"):
            raise ValueError(f"Invalid assets.images.canonical.format: {self.format!r}")
        if self.max_side is not None and self.max_side <= 0:
            raise ValueError("assets.images.canonical.max_side must be > 0 when set")
        if self.jpeg_quality < 1 or self.jpeg_quality > 100:
            raise ValueError("assets.images.canonical.jpeg_quality must be in [1, 100]")


@dataclass
class ImageRemoteConfig:
    """Remote fetch settings for image assets."""

    timeout_s: int = 20
    max_bytes: int = 20_000_000
    user_agent: str = "mmirage/0.x"

    def __post_init__(self) -> None:
        if self.timeout_s <= 0:
            raise ValueError("assets.images.remote.timeout_s must be > 0")
        if self.max_bytes <= 0:
            raise ValueError("assets.images.remote.max_bytes must be > 0")


@dataclass
class ImageAssetsConfig:
    """Top-level image asset settings."""

    root_dir: Optional[str] = None
    canonical: ImageCanonicalConfig = field(default_factory=ImageCanonicalConfig)
    remote: ImageRemoteConfig = field(default_factory=ImageRemoteConfig)


@dataclass
class AssetsConfig:
    """Asset managers configuration block."""

    images: ImageAssetsConfig = field(default_factory=ImageAssetsConfig)


@dataclass
class MMirageConfig:
    """Main configuration class for MMIRAGE pipeline.

    Contains all configuration needed to run a MMIRAGE processing pipeline,
    including processor configurations, dataset loading parameters, processing
    parameters, and execution parameters.

    Attributes:
        processors: List of processor configurations for data transformation.
        loading_params: Parameters for loading input datasets.
        processing_params: Parameters for processing dataset samples.
        execution_params: Parameters for executing the pipeline (local/SLURM).
    """

    processors: List[BaseProcessorConfig]
    loading_params: LoadingParams
    processing_params: ProcessingParams
    assets: AssetsConfig = field(default_factory=AssetsConfig)
    execution_params: ExecutionParams = field(default_factory=ExecutionParams)
