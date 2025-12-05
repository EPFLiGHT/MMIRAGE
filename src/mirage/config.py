from dataclasses import dataclass, field
from pydantic import BaseModel, create_model
from typing import Any, Dict, List, Literal, Optional, Type, cast

# Dataclasses

@dataclass
class EngineConfig:
    model_path: str
    tp_size: int = 4
    trust_remote_code: bool = True


@dataclass
class DatasetConfig:
    path: str
    type: Literal["JSONL", "loadable"] = "loadable"


@dataclass
class ProcessingGenParams:
    datasets: List[
        DatasetConfig
    ]  # One or more paths to HF datasets saved with 'save_to_disk'
    output_dir: str  # Root directory for shard outputs
    num_shards: int | str = (
        1  # Total number of shards (matches your sbatch array size).
    )
    shard_id: int | str = (
        0  # Index of this shard (0-based; usually $SLURM_ARRAY_TASK_ID).
    )
    conversations_field: str = (
        "conversations"  # Name of the column containing the list of dialog turns.
    )
    batch_size: int | str = 64  # Batch size for processing

    def __post_init__(self):
        if isinstance(self.num_shards, str):
            self.num_shards = int(self.num_shards) if self.num_shards.isdigit() else 1
        if isinstance(self.shard_id, str):
            self.shard_id = int(self.shard_id) if self.shard_id.isdigit() else 0
        if isinstance(self.batch_size, str):
            self.batch_size = int(self.batch_size) if self.batch_size.isdigit() else 64
        self.batch_size = max(self.batch_size, 1)

    def get_num_shards(self) -> int:
        return cast(int, self.num_shards)

    def get_shard_id(self) -> int:
        return cast(int, self.shard_id)

    def get_batch_size(self) -> int:
        return cast(int, self.batch_size)


@dataclass
class InputVar:
    name: str
    key: str


@dataclass
class OutputVar:
    name: str
    type: str
    output_type: Literal["plain", "JSON"]
    prompt: str
    output_schema: List[str] = field(
        default_factory=list
    )  # empty list if output_type is "plain"

    def get_output_schema(self) -> Optional[Type[BaseModel]]:
        if self.output_type == "JSON" and self.output_schema:
            fields: Dict[str, Any] = {
                var: (str, ...) for var in self.output_schema
            }  # ... means required field
            return create_model(f"OutputSchema", **fields)
        return None


@dataclass
class ProcessingParams:
    inputs: List[InputVar]
    outputs: List[OutputVar]
    output_schema: Dict[str, Any]


@dataclass
class MirageConfig:
    engine: EngineConfig
    sampling_params: Dict[str, Any]
    processing_gen_params: ProcessingGenParams
    processing_params: ProcessingParams
