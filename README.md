# MMIRAGE

MMIRAGE, which stands for Modular Multimodal Intelligent Reformatting and Augmentation Generation Engine, is an advanced platform designed to streamline the processing of datasets using generative models. It is engineered to handle large-scale data reformatting and augmentation tasks with efficiency and precision. By leveraging state-of-the-art generative models, MMIRAGE enables users to perform complex dataset transformations, ensuring compatibility across various formats and schemas. Its multi-node support and parallel processing capabilities make it an ideal choice for scenarios demanding substantial computational power, such as distributed training and inference workflows. MMIRAGE not only simplifies the integration of powerful language models but also provides a customizable framework for diverse use cases, from reformatting conversational datasets to generating Q/A pairs from plain text.

## How to install

To install the library, you can clone it from GitHub and then use pip to install it directly. It is recommended to have already installed `torch` and `sglang` to take advantage of GPU acceleration.

```bash
git clone git@github.com:EPFLiGHT/MMIRAGE.git
pip install -e ./MMIRAGE
```

For testing and scripts that make use of the library, it is advised to create a .env file. You can do this by running the following command:
```bash
curl https://raw.githubusercontent.com/EPFLiGHT/MMIRAGE/refs/heads/json-output/scripts/generate_env.sh | sh
```


## How to install

To install the library, you can clone it from GitHub and then use pip to install it directly. It is recommended to have already installed `torch` and `sglang` to take advantage of GPU acceleration.

```bash
git clone git@github.com:EPFLiGHT/MIRAGE.git
pip install -e ./MIRAGE
```

For testing and scripts that make use of the library, it is advised to create a .env file. You can do this by running the following command:
```bash
curl https://raw.githubusercontent.com/EPFLiGHT/MIRAGE/refs/heads/json-output/scripts/generate_env.sh | sh
```


## Key features

- Easily configurable with a YAML file which configure the following parameters
    - The prompt to the LLM
    - Variables with the name and their key to a JSON
- Parallelizable with a multi-node support
    - The training pipeline should use either distributed inference using accelerate 
- Support a variety of LLMs and VLMs (LLM only for a first version)
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

The dataset contains assistant answers that are badly formatted. The goal would be to use a LLM to format our answer in Markdown. With MMIRAGE, it would be as simple as defining a YAML configuration file.
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

## Usefool tools

- Jinja2 to process the YAML: #[link](https://jinja.palletsprojects.com/en/stable/)
- JMESPath: #[link](https://jmespath.org/)
- SGLang: #[link](https://github.com/sgl-project/sglang)
- Paper for performance drom: #[link](https://arxiv.org/abs/2408.02442)
