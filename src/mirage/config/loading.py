from dataclasses import dataclass, field
from typing import Union, List, cast

from mirage.core.loader.base import BaseDataLoaderConfig


@dataclass
class LoadingParams:
    datasets: List[BaseDataLoaderConfig] = field(default_factory=list)
    output_dir: str  = ""
    num_shards: Union[int, str] = 1
    shard_id: Union[int, str] = 0
    batch_size: Union[int, str] = 1  # Batch size for processing

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
        return cast(int, self.num_shards)

    def get_shard_id(self) -> int:
        return cast(int, self.shard_id)

    def get_batch_size(self) -> int:
        return cast(int, self.batch_size)


