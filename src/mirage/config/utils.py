from typing import Any, Dict, List, TypeAlias, Union, cast
from dacite import Config, from_dict
import yaml
import os

from mirage.config.config import MirageConfig
from mirage.core.process.base import BaseProcessorConfig, ProcessorRegistry, OutputVar
from mirage.core.loader.base import BaseDataLoaderConfig, DataLoaderRegistry

EnvValue: TypeAlias = Union[str, List["EnvValue"], Dict[str, "EnvValue"]]

def load_mirage_config(config_path: str) -> MirageConfig:
    """
    Load SGLang engine, sampling params, and batch size from YAML config.

    Example config:

    processors:
      - type: llm
        server_args:
          model_path: Qwen/Qwen3-4B-Instruct-2507
          tp_size: 4          # use all 4 GPUs on the node
          disable_custom_all_reduce: true
        sampling_params:
          temperature: 0.1
          top_p: 0.9
          max_new_tokens: 1024
          custom_params:
            chat_template_kwargs:
              enable_thinking: false

    loading_params:
      datasets:
        - path: tests/mock_data/data.jsonl
          type: JSONL
      output_dir: tests/output
      num_shards: 4
      shard_id: 0
      conversations_field: "conversations"
      batch_size: 64

    processing_params:
      inputs:
        - name: text
          key: text

      outputs:
        - name: formatted_answer
          type: llm
          output_type: JSON
          output_schema:
            - question
            - answer
          prompt: |
            Generate one question and its corresponding answer using the following text:
            ```
            {{ text }}
            ```

      remove_columns: True
      output_schema:
        conversations:
          - role: "user"
            content: "{{ formatted_answer.question }}"
          - role: "assistant"
            content: "{{ formatted_answer.answer }}"
    """

    with open(config_path, "r") as f:
        cfg: EnvValue = yaml.safe_load(f) or {}

    def expand_env_vars(obj: EnvValue) -> EnvValue:
        if isinstance(obj, dict):
            return {key: expand_env_vars(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [expand_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            return os.path.expandvars(obj)
        else:
            return obj

    def processor_config_hook(data: Dict[str, Any]) -> BaseProcessorConfig:
        clz = ProcessorRegistry.get_config_cls(data["type"])
        return from_dict(clz, data, config=config)

    def loader_config_hook(data: Dict[str, Any]) -> BaseDataLoaderConfig:
        clz = DataLoaderRegistry.get_config_cls(data["type"])
        return from_dict(clz, data, config=config)

    def output_var_hook(data: Dict[str, Any]) -> OutputVar:
        clz = ProcessorRegistry.get_output_var_cls(data["type"])
        return from_dict(clz, data, config=config)

    cfg = expand_env_vars(cfg)
    config = Config(
        type_hooks={
            BaseProcessorConfig: processor_config_hook,
            BaseDataLoaderConfig: loader_config_hook,
            OutputVar: output_var_hook,
        }
    )
    cfg_obj = from_dict(MirageConfig, cast(dict, cfg), config=config)

    return cfg_obj
