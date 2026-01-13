from typing import Any, Dict, List, TypeAlias, Union, cast
from dacite import Config, from_dict
import yaml
import os

from mirage.config.config import MirageConfig
from mirage.core.process.base import BaseProcessorConfig, ProcessorRegistry, Variable
from mirage.core.loader.base import BaseDataLoaderConfig, DataLoaderRegistry

EnvValue: TypeAlias = Union[str, List["EnvValue"], Dict[str, "EnvValue"]]


def load_mirage_config(config_path: str) -> MirageConfig:
    """
    Load SGLang engine, sampling params, and batch size from YAML config.

    Example config:

    engine:
      model_path: "meta-llama/Meta-Llama-3.1-8B-Instruct"

    sampling_params:
      temperature: 0.2
      top_p: 0.9

    processing_gen_params:
      datasets:
        - path: "/path/to/dataset1"
          type: loadable
        - path: "/path/to/dataset2.jsonl"
          type: JSONL
      output_dir: "/path/to/output"
      num_shards: 8
      shard_id: 0
      conversations_field: "conversations"

    processing_params:
      inputs:
        - name: assistant_answer
          key: conversations[1].content
        - name: user_prompt
          key: conversations[0].content
        - name: modalities
          key: modalities

      outputs:
        - name: formatted_answer
          type: llm
          output_type: plain
          prompt: |
            Reformat the answer in a markdown format without adding anything else:
            {assistant_answer}
          output_schema:
            - question
            - explanation
            - answer

      output_schema:
        conversations:
        - role: user
          content: {user_prompt}
        - role: assistant
          content: {formatted_answer}
        modalities: {modalities}
    """
    from mirage.config.config import MirageConfig

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

    def output_var_hook(data: Dict[str, Any]) -> Variable:
        clz = ProcessorRegistry.get_output_var_cls(data["type"])
        return from_dict(clz, data, config=config)


    cfg = expand_env_vars(cfg)
    config = Config(type_hooks={
        BaseProcessorConfig : processor_config_hook,
        BaseDataLoaderConfig : loader_config_hook,
        Variable : output_var_hook
    })
    cfg_obj = from_dict(MirageConfig, cast(dict, cfg), config=config)

    return cfg_obj



