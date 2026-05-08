<h1 align="center">

![image](https://raw.githubusercontent.com/EPFLiGHT/MMIRAGE/main/mmirage_logo_with_text.png)

</h1>

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

### Running (single command)

Run the pipeline via the CLI. Retry behavior is driven by your YAML config:

- `execution_params.retry: true` → automatically retries failed shards until completion or `max_retries`
- `execution_params.retry: false` → submits/runs once; you can later trigger retries via `check`
- `execution_params.merge: true` → after a successful run, automatically merges shard outputs

```bash
mmirage run --config configs/config_mock.yaml
```

To check status only:

```bash
mmirage check --config configs/config_mock.yaml
```

To check status and submit retries for failed shards:

```bash
mmirage check --config configs/config_mock.yaml --retry
```

To merge shards from the CLI directly:

```bash
mmirage merge --config configs/config_mock.yaml
```

To merge shards without a config file (input directory + output directory only):

```bash
mmirage merge-dir --input-dir /path/to/shards --output-dir /path/to/merged
```

`--input-dir` can point either to a single dataset directory that contains `shard_*`
folders, or to a parent directory containing multiple dataset subdirectories.
If `shard_*` folders are present directly in `--input-dir`, MMIRAGE merges that
root dataset directly and ignores nested internal folders.

For multiple datasets, you can also choose a shared merge root:

```bash
mmirage merge --config configs/config_mock.yaml --output-root /path/to/merged
```

MMIRAGE still keeps datasets separate by creating one subdirectory per dataset under the root.

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
  state_dir: /path/to/state/dir
  datasets:
    - path: /path/to/dataset
      type: loadable
      output_dir: /path/to/output/shards
  num_shards: 4
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

execution_params:
  mode: local
  retry: false
  merge: false
```

Configuration explanation:

- `processors`: List of processor configurations. Currently supports `llm` type for LLM-based generation.
- `loading_params`: Parameters for loading and sharding datasets.
  - `state_dir`: Optional shared directory for shard status/retry state. Defaults to `~/.cache/MMIRAGE/state_dir`.
  - `datasets`: List of dataset configurations with path, type, and output directory.
- `processing_params`:
  - `inputs`: Variables extracted from the input dataset using JMESPath queries.
  - `outputs`: Variables created by processors. Prompts use Jinja2 templating (`{{ variable }}`).
  - `output_schema`: Defines the structure of output samples.
- `execution_params`:
  - `mode`: "local" to run shard processing in the current Python environment or "slurm" to run through SLURM by submitting an sbatch array job.
  - `retry`: If true, MMIRAGE automatically retries failed shards until they succeed or `max_retries` is reached. If false, the pipeline runs/submits once, and retries can be triggered later via the check/retry CLI commands.
  - `merge`: If true, MMIRAGE merges shard outputs after a successful `run`. Merged datasets are written under each dataset `output_dir` in a `merged` subdirectory.

Merge output behavior with multiple datasets:
- Default (`run` with `execution_params.merge: true`, or `merge` without `--output-root`): each dataset is merged to its own `<dataset.output_dir>/merged`.
- Shared root (`merge --output-root ...`): one merged subdirectory is created per dataset under the root.

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
  state_dir: path/to/state/dir
  datasets:
    - path: /path/to/image/dataset
      type: loadable
      output_dir: /path/to/output/shards
  num_shards: 4
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

execution_params:
  mode: local
  retry: false
```

Key multimodal features:
- `chat_template`: Specify the VLM chat template (e.g., `qwen2-vl`)
- `type: image`: Mark input variables as images
- `image_base_path`: Base directory for resolving relative image paths
- Supports PIL Images, URLs, and file paths

### Benchmarking shard performance

Pass `--stats` to `run` or `submit` to enable per-shard benchmarking. This activates GPU
utilization polling and throughput tracking on compute nodes — disabled by default to
avoid unnecessary overhead.

```bash
# Local run with stats collection
mmirage run --config configs/config_mock.yaml --stats

```

After the run completes, inspect the results with:

```bash
mmirage stats --config configs/config_mock.yaml
```

This prints a JSON report with per-shard details and an aggregate summary:

```json
{
  "per_shard": [
    {
      "shard_id": 0,
      "status": "success",
      "started_at": "2026-04-30T10:00:00",
      "finished_at": "2026-04-30T10:01:05",
      "stats": {
        "runtime_seconds": 65.2,
        "runtime_human": "1m 5s",
        "rows_processed": 1024,
        "throughput_rows_per_sec": 15.7,
        "gpu_util_mean": 88.4,
        "gpu_util_min": 72.0,
        "gpu_util_max": 98.0,
        "gpu_util_samples": 13,
        "input_tokens": 512000,
        "output_tokens": 196608,
        "num_gpus": 4,
        "tokens_per_sec_per_gpu": 753.1,
        "gpu_days_per_billion_tokens": 0.0015
      }
    }
  ],
  "aggregate": {
    "total_shards": 1,
    "completed_shards": 1,
    "total_rows_processed": 1000,
    "wall_clock_runtime_seconds": 133.04,
    "wall_clock_runtime_human": "2m 13s",
    "sum_shard_runtime_seconds": 133.04,
    "sum_shard_runtime_human": "2m 13s",
    "min_shard_runtime_seconds": 133.04,
    "min_shard_runtime_human": "2m 13s",
    "max_shard_runtime_seconds": 133.04,
    "max_shard_runtime_human": "2m 13s",
    "overall_throughput_rows_per_sec": 7.52,
    "mean_gpu_util_pct": 86.2,
    "num_gpus": 4,
    "total_input_tokens": 146214,
    "total_output_tokens": 1022046,
    "sum_model_load_seconds": 38.272,
    "sum_inference_runtime_seconds": 94.768,
    "tokens_per_sec_per_gpu": 10784.72,
    "gpu_days_per_billion_tokens": 1.0732
  }
}
```

Key metrics:
- **`runtime_seconds`** / **`runtime_human`**: time from when the shard started on the cluster (after dispatch), excluding queue wait time.
- **`overall_throughput_rows_per_sec`**: total rows / wall-clock time across all shards running in parallel.
- **`mean_gpu_util_pct`**: mean percentage GPU utilization across shards.
- **`tokens_per_sec_per_gpu`**: output tokens generated per second per GPU — the primary throughput metric used by frameworks such as [DataTrove](https://github.com/huggingface/datatrove).
- **`gpu_days_per_billion_tokens`**: total GPU-days consumed to generate 1 billion output tokens — useful for cost and scaling comparisons across different hardware configurations.
- Token metrics are `null` when no LLM processor was active, and GPU stats are `null` when `nvidia-smi` is unavailable or `--stats` was not passed.

Reference benchmark:
- [DataTrove Benchmark](https://github.com/huggingface/datatrove/tree/main/examples/inference/benchmark)

The config `configs/config_benchmark_datatrove.yaml` mirrors the DataTrove inference benchmark conditions:

| Setting | Value |
|---|---|
| Dataset | `simplescaling/s1K-1.1` (train split, 1 000 samples) |
| Prompt | raw `question` field, no system prompt |
| Output | up to 1 024 tokens per sample |
| Context | 2 048-token model max context |
| Model | `Qwen/Qwen3-4B` (DataTrove baseline: tp=1 on a single GPU) |

Download the dataset before running:

```python
from datasets import load_dataset
ds = load_dataset('simplescaling/s1K-1.1', split='train')
ds.save_to_disk('data/s1K-1.1')
```

Then run with stats collection enabled:

```bash
mmirage run --config configs/config_benchmark_datatrove.yaml --stats
```

Inspect results:

```bash
mmirage stats --config configs/config_benchmark_datatrove.yaml
```

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
- DataTrove Benchmark: [link](https://github.com/huggingface/datatrove/tree/main/examples/inference/benchmark)
