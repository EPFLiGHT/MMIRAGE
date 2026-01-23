"""Data loading configuration for MIRAGE pipeline."""

from dataclasses import dataclass, field
from typing import Union, List, cast

from mirage.core.loader.base import BaseDataLoaderConfig


@dataclass
class LoadingParams:
    """Parameters for loading and distributing datasets across shards.

    Defines how datasets are loaded and processed in a distributed manner,
    supporting sharding for parallel processing.

    Attributes:
        datasets: List of dataset configurations to load.
        output_dir: Directory path for saving processed output shards.
        num_shards: Total number of shards to split the dataset into.
        shard_id: ID of this shard (0-indexed).
        batch_size: Batch size for processing samples.

    Raises:
        ValueError: If num_shards, shard_id, or batch_size cannot be converted to int.
    """

    datasets: List[BaseDataLoaderConfig] = field(default_factory=list)
    output_dir: str = ""
    num_shards: Union[int, str] = 1
    shard_id: Union[int, str] = 0
    batch_size: Union[int, str] = 1

    def __post_init__(self):
        if isinstance(self.num_shards, str):
            try:
                self.num_shards = int(self.num_shards)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid value for num_shards: {self.num_shards!r}")
        if isinstance(self.shard_id, str):
            try:
                self.shard_id = int(self.shard_id)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid value for shard_id: {self.shard_id!r}")
        if isinstance(self.batch_size, str):
            try:
                self.batch_size = int(self.batch_size)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid value for batch_size: {self.batch_size!r}")
        self.batch_size = max(self.batch_size, 1)

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
