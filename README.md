# MMIRAGE

MMIRAGE, which stands for **M**odular **M**ultimodal **I**ntelligent **R**eformatting and **A**ugmentation **G**eneration **E**ngine, is an advanced platform designed to streamline the processing of datasets using generative models, including vision-language models (VLMs). It is engineered to handle large-scale data reformatting and augmentation tasks with efficiency and precision. By leveraging state-of-the-art generative models, MMIRAGE enables users to perform complex dataset transformations, ensuring compatibility across various formats and schemas. Its multi-node support and parallel processing capabilities make it an ideal choice for scenarios demanding substantial computational power, such as distributed training and inference workflows. MMIRAGE not only simplifies the integration of powerful language models but also provides a customizable framework for diverse use cases, from reformatting conversational datasets to generating Q/A pairs from plain text.

## How to install

To install the library, you can clone it from GitHub and then use pip to install it directly. It is recommended to have already installed `torch` and `sglang` to take advantage of GPU acceleration.

```bash
git clone git@github.com:EPFLiGHT/MMIRAGE.git
pip install -e ./MMIRAGE
```

For testing and scripts that make use of the library, it is advised to create a .env file:
```bash
./scripts/generate_env.sh
```

## Key features

- **Multimodal Support**: Process both text and images with vision-language models
- Easily configurable with a YAML file which configures the following parameters:
    - The prompt to the LLM (using Jinja2 templating)
    - Variables with the name and their JMESPath key to a JSON
    - Image inputs for multimodal processing
- Parallelizable with multi-node support
    - The training pipeline uses distributed inference with sharding
- Support a variety of LLMs and VLMs (Vision-Language Models)
- Support any dataset schemas (configurable with the YAML format)
- The ability to either output a JSON (or any other structured format) or plain text
- Modular architecture with pluggable processors, loaders, and writers

## Example usage

### Text-only: Reformatting dataset

Suppose you have a dataset with samples of the following format

```json
{ 
    "conversations" : [{"role": "user", "content": "Describe the image"}, {"role": "assistant", "content": "This is a badly formmatted answer"}],
    "modalities" : ["<the images>"]
}
```

The dataset contains assistant answers that are badly formatted. The goal would be to use a LLM to format our answer in Markdown. With MMIRAGE, it would be as simple as defining a YAML configuration file:

```yaml
processors:
  - type: llm
    server_args:
      model_path: Qwen/Qwen3-8B
      tp_size: 4
      trust_remote_code: true
    default_sampling_params:
      temperature: 0.1
      top_p: 1.0
      max_new_tokens: 384

loading_params:
  datasets:
    - path: /path/to/dataset
      type: loadable
      output_dir: /path/to/output/shards
  num_shards: "$SLURM_ARRAY_TASK_COUNT"
  shard_id: "$SLURM_ARRAY_TASK_ID"
  batch_size: 64

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
        {{ assistant_answer }}
      
  remove_columns: false
  output_schema:
    conversations:
      - role: user
        content: "{{ user_prompt }}"
      - role: assistant
        content: "{{ formatted_answer }}"
    modalities: "{{ modalities }}"
```

Configuration explanation:

- `processors`: List of processor configurations. Currently supports `llm` type for LLM-based generation.
- `loading_params`: Parameters for loading and sharding datasets.
  - `datasets`: List of dataset configurations with path, type, and output directory.
- `processing_params`:
  - `inputs`: Variables extracted from the input dataset using JMESPath queries.
  - `outputs`: Variables created by processors. Prompts use Jinja2 templating (`{{ variable }}`).
  - `output_schema`: Defines the structure of output samples.

### Multimodal: Processing images with VLMs

MMIRAGE supports multimodal processing with vision-language models:

```yaml
processors:
  - type: llm
    server_args:
      model_path: Qwen/Qwen2-VL-7B-Instruct
      tp_size: 4
      trust_remote_code: true
    chat_template: qwen2-vl  # Required for VLMs
    default_sampling_params:
      temperature: 0.1
      top_p: 0.95
      max_new_tokens: 768

loading_params:
  datasets:
    - path: /path/to/image/dataset
      type: loadable
      output_dir: /path/to/output/shards
  num_shards: "$SLURM_ARRAY_TASK_COUNT"
  shard_id: "$SLURM_ARRAY_TASK_ID"
  batch_size: 32

processing_params:
  inputs:
    - name: medical_image
      key: image
      type: image  # Mark as image input
      image_base_path: /path/to/images  # Base directory for relative paths
    - name: original_caption
      key: caption
      type: text

  outputs:
    - name: enhanced_caption
      type: llm
      output_type: plain
      prompt: |
        Describe the medical image in detail.
        Original caption for context: {{ original_caption }}
        
  remove_columns: false
  output_schema:
    image: "{{ medical_image }}"
    caption: "{{ enhanced_caption }}"
    original_caption: "{{ original_caption }}"
```

Key multimodal features:
- `chat_template`: Specify the VLM chat template (e.g., `qwen2-vl`)
- `type: image`: Mark input variables as images
- `image_base_path`: Base directory for resolving relative image paths
- Supports PIL Images, URLs, and file paths

## Architecture

MMIRAGE uses a modular architecture:

```
mmirage/
├── config/           # Configuration loading and validation
├── core/
│   ├── loader/       # Dataset loaders (JSONL, HuggingFace)
│   ├── process/      # Processors (LLM, etc.) and variable system
│   │   └── processors/
│   │       └── llm/  # LLM processor with multimodal support
│   └── writer/       # Output rendering with Jinja2
├── shard_process.py  # Main processing script
└── merge_shards.py   # Shard merging utility
```

## Useful tools

- Jinja2 for template processing: [link](https://jinja.palletsprojects.com/en/stable/)
- JMESPath for JSON queries: [link](https://jmespath.org/)
- SGLang for fast inference: [link](https://github.com/sgl-project/sglang)
- Performance paper: [link](https://arxiv.org/abs/2408.02442)
