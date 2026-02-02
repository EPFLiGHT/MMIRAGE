# MIRAGE

MIRAGE, which stands for **M**ultimodal **I**ntelligent **R**eformatting and **A**ugmentation **G**eneration **E**ngine, is an advanced platform designed to streamline the processing of datasets using generative models, including vision-language models (VLMs). It is engineered to handle large-scale data reformatting and augmentation tasks with efficiency and precision. By leveraging state-of-the-art generative models, MIRAGE enables users to perform complex dataset transformations, ensuring compatibility across various formats and schemas. Its multi-node support and parallel processing capabilities make it an ideal choice for scenarios demanding substantial computational power, such as distributed training and inference workflows. MIRAGE not only simplifies the integration of powerful language models but also provides a customizable framework for diverse use cases, from reformatting conversational datasets to generating Q/A pairs from plain text.

## How to install

To install the library, you can clone it from GitHub and then use pip to install it directly. It is recommended to have already installed `torch` and `sglang` to take advantage of GPU acceleration.

```bash
git clone git@github.com:EPFLiGHT/MIRAGE.git
pip install -e ./MIRAGE
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
    "modalities" : [<the images>]
}
```

The dataset contains assistant answers that are badly formatted. The goal would be to use a LLM to format our answer in Markdown. With MIRAGE, it would be as simple as defining a YAML configuration file:

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

MIRAGE supports multimodal processing with vision-language models:

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

### Generating Q/A pairs from text

```yaml
processors:
  - type: llm
    server_args:
      model_path: Qwen/Qwen3-4B-Instruct
      tp_size: 1
    default_sampling_params:
      temperature: 0.1
      max_new_tokens: 1024

loading_params:
  datasets:
    - path: /path/to/text/dataset.jsonl
      type: JSONL
      output_dir: /path/to/output
  num_shards: 4
  shard_id: 0
  batch_size: 64

processing_params:
  inputs:
    - name: plain_text
      key: text
    
  outputs:
    - name: qa_pair
      type: llm
      output_type: JSON  # Structured JSON output
      output_schema:
        - question
        - answer
      prompt: | 
        Generate one question and its answer from this text:
        {{ plain_text }}
        
  remove_columns: true
  output_schema:
    conversations:
      - role: "user"
        content: "{{ qa_pair.question }}"
      - role: "assistant"
        content: "{{ qa_pair.answer }}"
```

## Architecture

MIRAGE uses a modular architecture:

```
mirage/
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
  conversations:
    - role: user
      content: {question}
    - role: assistant
      content: |
        {explanation}
        Answer: {answer}

```

Here, we choose to output a JSON answer with 3 keys ("question", "explanation" and "answer"). That we will match

### Working with Images (Multimodal)

MIRAGE supports Vision-Language Models (VLMs) for processing datasets that contain images. It handles two common scenarios:

**Scenario 1: Embedded Images**  
Dataset has actual image objects (PIL Images) in the columns. HuggingFace automatically decodes these when loading datasets with `Image` feature types.

**Scenario 2: Path-Based Images**  
Dataset contains image filenames/paths that reference external image files (like PMC-OA with `images.zip`).

**Example: Path-based dataset (PMC-OA)**

Suppose you have a medical imaging dataset with the following format:

```json
{
    "image": "PMC212319_Fig3_4.jpg",
    "caption": "A. Real time image of the translocation of ARF1-GFP to the plasma membrane..."
}
```

The images are stored separately (e.g., in an extracted `images/` folder). Configure MIRAGE like this:

```yaml
engine:
  model_path: Qwen/Qwen2.5-VL-7B-Instruct  # Vision-language model
  chat_template: qwen2-vl  # Chat template for vision-language models (defaults to "qwen2-vl")

inputs:
  - name: medical_image
    key: image
    type: image  # Indicates this is an image input
    image_base_path: /path/to/images  # Base directory where image files are stored
  - name: original_caption
    key: caption
    type: text

outputs:
  - name: enhanced_caption
    type: llm
    output_type: plain
    prompt: |
      You are a medical imaging expert. Analyze the provided medical image and enhance the caption.
      
      Original caption: {original_caption}
      
      Provide a more detailed and accurate caption based on what you see in the image.

output_schema:
  image: "{medical_image}"  # Image passed through unchanged
  caption: "{enhanced_caption}"
  original_caption: "{original_caption}"
```

**Example: Embedded images (HuggingFace datasets with Image feature)**

For datasets where images are already embedded as PIL Images:

```yaml
inputs:
  - name: photo
    key: image
    type: image  # No image_base_path needed - images are already loaded
  - name: caption
    key: caption
```

**Important notes:**
- Images are **never modified** - they are passed through to the output unchanged
- Use `image_base_path` only for path-based datasets where images are stored separately
- Supports file paths, URLs, PIL Images, and other formats accepted by SGLang
- See [SGLang supported VLMs](https://docs.sglang.io/supported_models/multimodal_language_models.html) for compatible models
- The model must be a Vision-Language Model to process images
- **Chat template**: Specify the appropriate chat template in the engine config (e.g., `chat_template: qwen2-vl`). Defaults to "qwen2-vl" if not specified

## Useful tools

- Jinja2 to process the YAML: #[link](https://jinja.palletsprojects.com/en/stable/)
- JMESPath: #[link](https://jmespath.org/)
- SGLang: #[link](https://github.com/sgl-project/sglang)
- Paper for performance drom: #[link](https://arxiv.org/abs/2408.02442)