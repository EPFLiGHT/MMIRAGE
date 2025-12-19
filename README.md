# MIRAGE

MIRAGE, which stands for Multimodal Intelligent Reformatting and Augmentation Generation Engine, is an advanced platform designed to streamline the processing of datasets using generative models. It is engineered to handle large-scale data reformatting and augmentation tasks with efficiency and precision. By leveraging state-of-the-art generative models, MIRAGE enables users to perform complex dataset transformations, ensuring compatibility across various formats and schemas. Its multi-node support and parallel processing capabilities make it an ideal choice for scenarios demanding substantial computational power, such as distributed training and inference workflows. MIRAGE not only simplifies the integration of powerful language models but also provides a customizable framework for diverse use cases, from reformatting conversational datasets to generating Q/A pairs from plain text.

## Key features

- Easily configurable with a YAML file which configure the following parameters
    - The prompt to the LLM
    - Variables with the name and their key to a JSON
    - Image inputs for multimodal processing
- Parallelizable with a multi-node support
    - The training pipeline should use either distributed inference using accelerate 
- Support a variety of LLMs and VLMs (Vision-Language Models)
- Support any dataset schemas (configurable with the YAML format)
- The ability to either output a JSON (or any other structured format) or a plain text

## Example usage

### Reformatting dataset

Suppose you have a dataset with samples of the following format

```json
{ 
    "conversations" : [{"role": "user", "content": "Describe the image"}, {"role": "assistant", "content": "This is a badly formmatted answer"}],
    "modalities" : [<the images>]
}
```

The dataset contains assistant answers that are badly formatted. The goal would be to use a LLM to format our answer in Markdown. With MIRAGE, it would be as simple as defining a YAML configuration file.
Then in the YAML configuration file, we could specify

```yaml
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
  conversations:
    - role: user
      content: {user_prompt}
    - role: assistant
      content: {formatted_answer}
  modalities: {modalities}

```

Configuration explanation:

- `inputs`: specify variables that are defined from the input dataset. For instance by specifying the key `conversations[1].content`, we say that this variable corresponds to `sample["conversations"][1]["content"]`
- `outputs`: specify variables that are created from the pipeline. We specify how the variable should be created: 
    - Here `formatted_answer` is created using a LLM prompt and is a plain text variable (as opposed to JSON variables)
- `output_schema`: specify the output schema of the dataset. So each sample will follow this format. Here we know that each sample will contain 2 keys: `conversations` and `modalities`

### Transforming datasets

In the second example, we want to generate questions from plain text document. The 3 keys that we want to generate are:

- "question"
- "answer"
- "explanation"

Suppose we have the following format:

```json
{
    "text" : "This is a very interesting article about cancer"
}
```

```yaml
inputs:
  - name: plain_text
    key: text
    
outputs:
  - name: output_dict
    type: prompt
    output_type: JSON
    prompt: | 
      I want to generate Q/A pairs from the following text:
      {plain_text}
    output_schema:
      - question
      - explanation
      - answer
        
output_schema:
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
  # chat_template: qwen2-vl  # Optional: explicitly specify chat template (auto-inferred if not set)

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
- **Chat template**: The chat template is automatically inferred from the model path (e.g., Qwen models use "qwen2-vl"). You can explicitly set it in the engine config with `chat_template: <template_name>` if needed

## Useful tools

- Jinja2 to process the YAML: #[link](https://jinja.palletsprojects.com/en/stable/)
- JMESPath: #[link](https://jmespath.org/)
- SGLang: #[link](https://github.com/sgl-project/sglang)
- Paper for performance drom: #[link](https://arxiv.org/abs/2408.02442)