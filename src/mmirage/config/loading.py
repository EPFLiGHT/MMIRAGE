"""Data loading configuration for MMIRAGE pipeline."""

import os
import re
from dataclasses import dataclass, field
from typing import Union, List, cast

from mmirage.core.loader.base import BaseDataLoaderConfig

DEFAULT_STATE_DIR = "~/.cache/MMIRAGE/state_dir"


@dataclass
class LoadingParams:
    """Parameters for loading and distributing datasets across shards.

    Defines how datasets are loaded and processed in a distributed manner,
    supporting sharding for parallel processing.

    Attributes:
        datasets: List of dataset configurations to load.
        state_dir: Shared directory for logical shard state/markers/retry tracking.
        output_dir: Legacy top-level output directory. Prefer per-dataset output_dir.
        num_shards: Total number of shards to split the dataset into.
        shard_id: ID of this shard (0-indexed).
        batch_size: Batch size for processing samples.

    Raises:
        ValueError: If num_shards, shard_id, or batch_size cannot be converted to int.
    """

    datasets: List[BaseDataLoaderConfig] = field(default_factory=list)
    state_dir: str = DEFAULT_STATE_DIR
    output_dir: str = ""
    num_shards: Union[int, str] = 1
    shard_id: Union[int, str] = 0
    batch_size: Union[int, str] = 1

    def __post_init__(self):
        _UNRESOLVED_ENV_VAR_PATTERN = re.compile(r"^\$(?:\{[A-Za-z_][A-Za-z0-9_]*\}|[A-Za-z_][A-Za-z0-9_]*)$")
        def is_unresolved_env_var(s: str) -> bool:
            return bool(_UNRESOLVED_ENV_VAR_PATTERN.fullmatch(s.strip()))
        
        if isinstance(self.num_shards, str):
            try:
                self.num_shards = int(self.num_shards)
                if self.num_shards < 1:
                    raise ValueError()
            except (ValueError, TypeError):
                if is_unresolved_env_var(self.num_shards):
                    self.num_shards = 1
                else:
                    raise ValueError(f"Invalid value for num_shards: {self.num_shards!r}")

        if isinstance(self.shard_id, str):
            try:
                self.shard_id = int(self.shard_id)
            except (ValueError, TypeError):
                if is_unresolved_env_var(self.shard_id):
                    self.shard_id = 0
                else:
                    raise ValueError(f"Invalid value for shard_id: {self.shard_id!r}")

        if isinstance(self.batch_size, str):
            try:
                self.batch_size = int(self.batch_size)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid value for batch_size: {self.batch_size!r}")

        self.batch_size = max(self.batch_size, 1)

        raw_state_dir = "" if self.state_dir is None else str(self.state_dir)
        self.state_dir = raw_state_dir.strip()
        if not self.state_dir:
            self.state_dir = DEFAULT_STATE_DIR

        self.state_dir = os.path.expanduser(self.state_dir)

    def get_state_root(self) -> str:
        """Get the state root path.

        Returns:
            str: State root path.
        """
        return self.state_dir

    def get_num_shards(self) -> int:
        """Get the total number of shards.

        Returns:
            int: Total number of shards.
        """
        return cast(int, self.num_shards)

    def get_shard_id(self) -> int:
        """Get the ID of this shard.

        Returns:
            int: Shard ID (0-indexed).
        """
        return cast(int, self.shard_id)

    def get_batch_size(self) -> int:
        """Get the batch size for processing.

        Returns:
            int: Batch size (minimum 1).
        """
        return cast(int, self.batch_size)
