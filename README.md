# 🐱 Groq QA Generator
[![Documentation](https://img.shields.io/badge/Documentation-available-brightgreen)](https://jcassady.github.io/groq-qa-generator)
![Issues](https://img.shields.io/github/issues/jcassady/groq-qa-generator?logo=github&label=Issues)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/a6129ef346ce412e953cc7949cee1599)](https://app.codacy.com/gh/jcassady/groq-qa-generator?utm_source=github.com&utm_medium=referral&utm_content=jcassady/groq-qa-generator&utm_campaign=Badge_Grade)
![Pytests](https://github.com/jcassady/groq-qa-generator/actions/workflows/run-pytest.yml/badge.svg)
![License](https://img.shields.io/github/license/jcassady/groq-qa-generator?label=License)
![Contributors](https://img.shields.io/github/contributors/jcassady/groq-qa-generator?logo=githubsponsors&label=Contributors)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

![GitHub release](https://img.shields.io/github/v/release/jcassady/groq-qa-generator?label=Release)
![Last commit](https://img.shields.io/github/last-commit/jcassady/groq-qa-generator?logo=github&label=Last%20commit&color=blue)
![GitHub repo size](https://img.shields.io/github/repo-size/jcassady/groq-qa-generator?logo=github&label=Repo%20size)
![Python versions](https://img.shields.io/pypi/pyversions/groq-qa-generator)
[![Downloads](https://static.pepy.tech/badge/groq-qa-generator)](https://pepy.tech/project/groq-qa-generator)
![Visitors](https://img.shields.io/badge/Visits-11663-blue)

<p align="center">
  <img src="./assets/images/logo.png" alt="Groqqy" /><br>
  <strong>Q</strong>: <em>“Would you tell me, please, which way I ought to go from here?”</em><br>
  <strong>A</strong>: <em>“That depends a good deal on where you want to get to,” said the Cat.</em><br>
  <em>— <strong>Alice's Adventures in Wonderland</strong></em>
</p>


|  <p align="left">**Groq QA** is a Python library that automates creating question-answer pairs from text to fine-tune large language models (LLMs). Powered by **[Groq](https://groq.com/)** and extended with **[Hugging Face](https://huggingface.co/)**, it uses models like **[LLaMA 3](https://www.llama.com/)** (70B parameters, 128K tokens) to generate high-quality QA pairs. The tool streamlines dataset preparation and enables easy fine-tuning in research, education, and domain-specific applications.</p> |
|---------------------------------------------------------------------------------------------------------------|
| <p align="center">**Note**: ***This project is not affiliated with or endorsed by Groq, Inc.***</p> |





## ✨ Features
|   | ✨ Feature                         | 📄 Description                                                             |
|----|-------------------------------------|-----------------------------------------------------------------------------|
| ✅ | **CLI**                             | Use the CLI tool with the `groq-qa` command.                                                 |
| ✅ | **Python Library**          | Import directly to your own Python project and extend your code.                                             |
| ✅ | **Advanced Models**                   | Supports large models like **LLaMA 3.1 70B** via the Groq API.               |
| ✅ | **Automated QA Generation**         | Generate question-answer pairs from input text.                             |
| ✅ | **Hugging Face Datasets**    | Convert QA pairs to Hugging Face datasets, then export or upload datasets to Hugging Face. |
| ✅ | **Custom Split Ratios**              | Define custom train/test split ratios when generating QA pairs. |
| ✅ | **Prompt Templates**                | Flexible question generation through prompt templates.                      |
| ✅ | **Customizable Configurations**     | Configure via CLI, `config.json`, or directly in Python code.                              |

## 👨‍💻 About the Developer
Hey there! I’m **[Jordan](https://jordan.cassady.me/)**, a Canadian network engineer with over a decade of experience, especially from my time in the fast-paced world of California startups. ✌️ My focus has been on automating test systems aligned with company KPIs, making it easier for teams to make data-driven decisions.

Whether it’s tackling tough challenges, improving codebases, or working on innovative ideas, I’m always up for the task. Let’s **[connect on LinkedIn](https://www.linkedin.com/in/jordan-cassady)** and make things happen! 🤝
  
[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/F2F814ELQM)


## 📄 Table of Contents
- [🐱 Groq QA Generator](#-groq-qa-generator)
- [✨ Features](#-features)
- [👨‍💻 About the Developer](#-about-the-developer)
- [🚀 Quick Start](#-quick-start)
- [📦 Upgrading](#-upgrading)
- [📖 Documentation](#-documentation)
  - [Documentation Generation](#documentation-generation)
- [⚙️ Using groq-qa](#-using-groq-qa)
  - [Setup the API Keys](#setup-the-api-keys)
  - [Setting the Environment Variables](#setting-the-environment-variables)
    - [Option 1: Using a `.env` File](#option-1-using-a-env-file-recommended)
    - [Option 2: Exporting Directly in the Terminal](#option-2-exporting-directly-in-the-terminal)
  - [Command-Line Interface (CLI)](#command-line-interface-cli)
    - [CLI Options](#cli-options)
- [🤗 Hugging Face Datasets](#-hugging-face-datasets)
  - [Export and Upload](#export-and-upload)
- [🛠 Configuration](#-configuration)
  - [Directory Structure](#directory-structure)
  - [Default config.json](#default-configjson)
- [🔧 Customizing the Configuration](#-customizing-the-configuration)
- [🐇 Input Data](#-input-data)
  - [Sample Input Data](#sample-input-data)
- [🤖 Models](#-models)
  - [Example](#example)
- [🧠 Technical Overview](#-technical-overview)
- [🧪 Testing Overview](#-testing-overview)
  - [🥼 Running Tests](#-running-tests)
- [🤝 How to Contribute](#-how-to-contribute)
- [❓ FAQ](#-faq)
- [⚖️ License](#-license)


## 🚀 Quick Start

1. Install the package via `pip`:
    ```bash
    pip install groq-qa-generator
    ```

2. Set up the API key:
    ```bash
    export GROQ_API_KEY=your_api_key_here
    ```

3. Run the `groq-qa` command with default settings (`~/.groq_qa/config.json`):
    ```bash
    groq-qa
    ```
4. View the results in `~/.groq_qa/qa_output.txt`.

## 📦 Upgrading

To ensure that you have the latest features, bug fixes, and improvements, it is recommended to periodically upgrade the `groq_qa_generator` package.

You can update `groq_qa_generator` to the latest version by running:

```bash
pip install --upgrade groq-qa-generator
```

## 📖 Documentation
You can access the full HTML documentation here:

👉 [**Groq QA Generator Documentation**](https://jcassady.github.io/groq-qa-generator/) 👈

### Documentation Generation
The documentation is automatically generated using **Sphinx**, a documentation generation tool for Python projects. Every change made to the documentation directory (`docs/`) triggers a GitHub Actions workflow that builds the HTML files and deploys them to **GitHub Pages**. This ensures that the documentation stays up-to-date with the latest project changes.


## ⚙️ Using groq-qa

### Setup the API Keys
First, you need to acquire both a **Groq API key** and a **Hugging Face token**.

- To get the **Groq API key**, sign up at the [Groq](https://groq.com) website and follow their instructions for generating a key.
- To obtain the **Hugging Face token**, create an account on [Hugging Face](https://huggingface.co/) and generate a token in your account settings.

#### Setting the Environment Variables
##### **Option 1**: Using a `.env` File (Recommended)

1. Create a `.env` file in your home directory (`~/`) with the following content:
   ```bash
   export GROQ_API_KEY=gsk_example_1234567890abcdef
   export HF_TOKEN=hf_example_1234567890abcdef
2. Source the `.env` file to load the environment variables:

   ```bash
   source ~/.env
##### **Option 2**: Exporting Directly in the Terminal
Alternatively, you can export the key and token directly in your terminal.
```bash
export GROQ_API_KEY=gsk_example_1234567890abcdef
export HF_TOKEN=hf_example_1234567890abcdef
```

### Command-Line Interface (CLI)

Once installed, the command `groq-qa` becomes available. By default, this command reads from the default configuration located at `~/.groq_qa/config.json`. 

Here are a few examples of how to use the `groq-qa` command:

```bash
# Run with default config.json:
groq-qa 

# Output results in JSON format:
groq-qa --json 

# Run with model and temperature overrides:
groq-qa --model llama3-70b-8192 --temperature 0.9

# Run with questions override:
groq-qa --questions 1

# Run with a custom train/test split of 70% training data and JSON output:
groq-qa --split 0.7 --json

# Upload generated QA pairs to Hugging Face:
groq-qa --upload example-username/example-dataset-name
```

#### CLI Options:
* 🧠 `--model`: The default model to be used for generating QA pairs is defined in `config.json`. The default is set to `llama3-70b-8192`.
* 🔥 `--temperature`: Controls the randomness of the model's output. Lower values will result in more deterministic and focused outputs, while higher values will make the output more random. The default is set to `0.1`.
* ❓ `--questions`: Allows you to specify the exact number of question-answer pairs to generate per chunk of text. For example, using `1` will force the system to generate 1 QA pair for each chunk, regardless of chunk size or token limits.
* 🗃️ `--json`: If this flag is included, the output will be saved in a JSON format. By default, the output is stored as a plain text file. The default is set to `False`.
* 📤 `--upload`: Hugging Face repository path for uploading the QA dataset. For example, `example-username/example-dataset-name`.
* ✂️ `--split`: Fraction of the dataset to be used for training. For example, `0.8` will allocate **80%** of the data for training and **20%** for testing.

**Note**: You can print out the full list of available CLI options and arguments by using the `--help` option:
```bash
groq-qa --help
```

## 🤗 Hugging Face Datasets

`groq-qa-generator` allows you to export the generated question-answer pairs in JSON format and upload them to [Hugging Face](https://huggingface.co/) as a dataset. This functionality enables you to easily integrate the generated QA pairs into machine learning pipelines, making it ready for fine-tuning models.


### Export and Upload:
1. Generate the QA pairs and export them in JSON format using the `--json` flag:
    ```bash
    groq-qa --json
    ```

2. Use the `--upload` flag to specify the Hugging Face repository where the dataset should be uploaded:
    ```bash
    groq-qa --json --upload your-huggingface-username/your-dataset-repo
    ```

3. Optionally, you can also specify a train/test split ratio using the `--split` argument. The default is to split the data **80%** for training and **20%** for testing:
    ```bash
    groq-qa --json --upload your-huggingface-username/your-dataset-repo --split 0.8
    ```

The dataset will be uploaded to Hugging Face in the `DatasetDict` format, and you can view or further process the dataset in your Hugging Face account.


## 🛠 Configuration
When you run the `groq-qa` command for the first time, a user-specific configuration directory (`~/.groq_qa/`) is automatically created. This directory contains all the necessary configuration files and templates for customizing input, prompts, and output.

### Directory Structure
```bash
~/.groq_qa/
├── config.json
├── data
│   ├── alices_adventures_in_wonderland.txt
│   ├── sample_input_data.txt
│   └── prompts
│       ├── sample_question.txt
│       └── system_prompt.txt
├── qa_output_training.json
├── qa_output_test.json
└── qa_output_training.txt
└── qa_output_test.txt
```

### Default config.json
```json
{
    "system_prompt": "system_prompt.txt",
    "sample_question": "sample_question.txt",
    "input_data": "sample_input_data.txt",
    "output_file": "qa_output.txt",
    "split_ratio": 0.8,
    "huggingface_repo": "username/dataset",
    "model": "llama3-70b-8192",
    "chunk_size": 512,
    "tokens_per_question": 60,
    "temperature": 0.1,
    "max_tokens": 1024
}
```

### 🔧 Customizing the Configuration

The `~/.groq_qa` directory contains essential files that can be customized to suit your specific needs. This directory includes the following components:

- 📄 **config.json**: This is the main configuration file where you can set various parameters for the QA generation process. You can customize settings such as:
  - 📝 **system_prompt**: Specify the path to your custom system prompt file that defines how the model should behave.
  - ❓ **sample_question**: Provide the path to a custom sample question file that helps guide the generation of questions.
  - 📖 **input_data**: Set the path to your own text file from which you want to generate question-answer pairs.
  - 💾 **output_file**: Define the path where the generated QA pairs will be saved.
  - 📊 **split_ratio**: Specify the ratio of the dataset to be used for training. The default is `0.8` for **80%** training and **20%** testing.
  - 🤗 **huggingface_repo**: Set the Hugging Face repository repo where the dataset will be uploaded.

Other configurable options include:
- 🤖 **model**: Select the model to be used for generating QA pairs (e.g., `llama3-70b-8192`).
- 📏 **chunk_size**: Set the number of tokens for each text chunk (e.g., `512`).
- 🪙 **tokens_per_question**: Specify the number of tokens allocated for each question (e.g., `60`).
- 🔥 **temperature**: Control the randomness of the model's output (e.g., `0.1`).
- 🪙 **max_tokens**: Define the maximum number of tokens the model can generate in the response (e.g., `1024`).

By adjusting these files and settings, you can create a personalized environment for generating question-answer pairs that align with your specific use case.


## 🐇 Input Data 

This project uses text data from *Alice's Adventures in Wonderland* by Lewis Carroll, sourced from [Project Gutenberg](https://www.gutenberg.org/). The full text is available in the included `data/alices_adventures_in_wonderland.txt` file.

### Sample Input Data

For demonstration purposes, a smaller sample of the full text is included in `data/sample_input_data.txt`. This file contains a portion of the main text, used to quickly test and generate question-answer pairs without processing the entire book.


## 🤖 Models
The `groq_qa_generator` currently supports the following models via the Groq API:


| Model Name           | Model ID                     | Developer | Context Window       | Description                                                                                       | Documentation Link                                                                 |
|----------------------|------------------------------|-----------|----------------------|---------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| **LLaMA 70B**        | llama3-70b-8192             | Meta      | 8,192 tokens         | A large language model with 70 billion parameters, suitable for high-quality QA pair generation. | [Model Card](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)               |
| **LLaMA 3.1 70B**    | llama-3.1-70b-versatile     | Meta      | 128k tokens          | A versatile large language model suitable for diverse applications.                               | [Model Card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md) |

**Note:** For optimal QA pair generation, it is recommended to use a larger model such as 70B, as its capacity helps ensure higher quality output. See Groq's [supported models](https://console.groq.com/docs/models) documentation for all options.

### 🐍 Python Library

In addition to CLI usage, the `groq_qa_generator` can be used directly in your Python project. Below is an example of how to configure and execute the question-answer generation process using a custom configuration:

#### Example

```python
# main.py

from groq_qa_generator import groq_qa

def main():
  # Define a custom configuration
    custom_config = {
        "system_prompt": "custom_system_prompt.txt",
        "sample_question": "custom_sample_question.txt",
        "input_data": "custom_input_data.txt",
        "output_file": "qa_output.txt",
        "split_ratio": 0.8,
        "huggingface_repo": "username/dataset",
        "model": "llama3-70b-8192",
        "chunk_size": 512,
        "tokens_per_question": 60,
        "temperature": 0.1,
        "max_tokens": 1024
    }

# Generate question-answer pairs
train_dataset, test_dataset = groq_qa.generate(custom_config)

# Print both train and test datasets
print(f"Train Dataset: {train_dataset}"
      f"\n\nTest Dataset: {test_dataset}")

if __name__ == "__main__":
    main()
```
This allows you to integrate the functionality within any Python application easily.


## 🧠 Technical Overview
1. 🔑 **API Interaction**:
    - **API Keys**: API keys and security tokens are securely retrieved from environment variables to ensure safe access to the Groq and Hugging Face API.
      - **Groq API**: A Groq client is initialized to enable communication, providing access to powerful models like **LLaMA 70B**.
      - **Hugging Face API**: Hugging Face Hub is used for uploading the generated QA datasets directly to a repository. This allows for easy integration of datasets into the Hugging Face ecosystem, facilitating model fine-tuning and further dataset management.


2. 📄 **Text Processing**:
    - **Loading Prompts and Questions**: The library includes methods to load sample questions and system prompts from specified file paths. These prompts are essential for guiding **LLaMA 70B**'s response.
    - **Generating Full Prompts**: The system prompt and sample question are combined into a complete prompt for the Groq API.

3. 🤖 **QA Pair Generation**:
    - The core process involves taking a list of text chunks and generating question-answer pairs using the **Groq** API to prompt **LLaMA 70B**.
        - Loads the system prompt and sample question.
        - Iterates through each text chunk, creating a full prompt for **LLaMA 70B**.
        - Retrieves the completion from the **Groq** API, and in turn the model.
        - Streams the completion response and converts it into question-answer pairs.
        - Splits the QA pairs into separate train and test files based on the split ratio (default is 80% train, 20% test).
        - Converts the generated QA pairs into a **Hugging Face** dataset and optionally uploads to a repository.


## 🧪 Testing Overview

**groq-qa-generator** includes comprehensive tests that ensure its core functionalities are reliable and efficient:

- **API Interactions**: Tests like `test_groq_api.py` mock API calls (e.g., `get_groq_client()`) to verify that external APIs such as **Groq** and **Hugging Face** function correctly. This ensures robust behavior even in different environments.
  
- **Configuration Handling**: Tests in `test_config.py` check proper loading of user-defined and default settings, ensuring flexibility and consistency across deployments.

- **Tokenization & Data Processing**: `test_tokenizer.py` and `test_dataset_processor.py` verify accurate text tokenization and the conversion of QA pairs into usable datasets. This is crucial for generating high-quality outputs.

- **QA Generation**: Core tests in `test_qa_generation.py` ensure reliable generation of question-answer pairs, which is the main functionality of the library.

- **Logging**: `test_logging_setup.py` ensures proper logging configuration, aiding in debugging and performance tracking.


### 🥼 Running Tests

To run the project's tests, you can use [Poetry](https://python-poetry.org/) and `pytest`. Follow these steps:

1. 📦 **Install Poetry**: If you haven't already, install Poetry using pip.
   ```bash
   pip install poetry
   ```
2. 🔧 **Install Dependencies**: Navigate to the project directory and install the dependencies.
   ```bash
   cd groq-qa-generator
   poetry install
   ```
3. ⚙️ **Confirm the Environment**: Verify that the virtual environment has been correctly set up and activated.
   ```bash
   poetry shell
   ```
4. 🏃 **Run Tests**: Use pytest to run the tests.
   ```bash
   poetry run pytest
   ```


## 🤝 How to Contribute

1. 🍴 **Fork the Repository**: Click the "Fork" button at the top-right of the repository page to create your copy.

2. 📥 **Clone Your Fork**: Clone the forked repository to your local machine.
   ```bash
   git clone https://github.com/your-username/groq-qa-generator.git
   ```
3. 🌿 **Create a Branch**: Use a descriptive name for your branch to clearly indicate the purpose of your changes. This helps maintain organization and clarity in the project.
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. 🔧 **Set Up the Environment**: Use Poetry to install the project dependencies.
   ```bash
   cd groq-qa-generator
   poetry install
   ```
5. ⚙️ **Confirm the Environment**: Verify that the virtual environment has been correctly set up and activated.
   ```bash
   poetry shell
   ```
6. 📦 **List Installed Packages**: Ensure that the dependencies have been installed correctly by listing the installed packages.
   ```bash
   poetry show
7. 📝 **Pick an Existing Issue or Suggest a New One**:
   - Check out the [issues page](https://github.com/jcassady/groq-qa-generator/issues) to see if there's an open issue you’d like to work on. If you find one, just drop a comment to let everyone know you're taking it on.
   - If you don’t see anything related to what you're working on, feel free to [create a new issue](https://github.com/jcassady/groq-qa-generator/issues/new) to describe the bug, feature, or improvement you have in mind.

8. 🔄 **Commit and Push**: After making your changes, commit them with a clear message and push your branch to your forked repository.
   ```bash
   git add .
   git commit -m "Add a concise description of your changes"
   git push origin feature/your-feature-name
   ```

## ❓ FAQ
### 📁 Where can I find the generated QA pairs?
The generated QA pairs are saved to the `output_file` defined in your `config.json` file. By default, it saves the output in `qa_output.txt`, located in your home directory’s `.groq_qa` folder (`~/.groq_qa/qa_output.txt`).

To change the output file name, edit the output_file field in your `config.json` file:

```bash
{
    "output_file": "qa_custom_output.txt"
}
```


### 🛠 Can I modify the sample question or system prompt templates?
Yes, both the system prompt and the sample question can be modified. These templates are located in the prompts directory inside the `~/.groq_qa/` folder:

* `system_prompt.txt`: Defines how the model should behave and guide the generation process.
* `sample_question.txt`: Defines how sample questions should be structured.
Feel free to edit these templates to suit your needs.

### 🔄 How do I override default configuration settings?
You can override the default configuration settings in two ways:
1. Edit the `config.json` file located in the `~/.groq_qa/` directory.
2. Pass command-line arguments to override specific settings, for example:

```bash
groq-qa --model llama3-70b-8192 --temperature 0.9 --json
```

### 🎛 How can I increase the randomness of the output?
Increase the `temperature` value in the configuration or pass it as a command-line argument (e.g., `--temperature 0.9`).

### 🔍 How do I upload my QA pairs to Hugging Face?
The tool allows you to upload QA pairs directly to Hugging Face. You can configure the Hugging Face repository in the `config.json` file under the `huggingface_repo` field or pass it as a command-line argument using the `--upload` option.

### 🔢 How can I split the dataset into training and test sets?
You can split your QA pairs dataset using the **Hugging Face** integration by specifying a split ratio in the configuration file or passing it via the command line with the `--split` argument. For example, to split **80%** of the data for training and **20%** for testing, use `--split 0.8`.

### 🐍 Can I use this tool in a larger Python project?
Yes, `groq_qa_generator` can be used as a Python library within your project. Simply import the module and configure it to generate QA pairs programmatically.

### 🌱 How can I contribute to the project?
If you'd like to contribute, feel free to browse the [issues page](https://github.com/jcassady/groq-qa-generator/issues) to find something to work on or propose a new issue. Fork the repository, create a new branch, and submit a pull request once your changes are ready!

## ⚖️ License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
