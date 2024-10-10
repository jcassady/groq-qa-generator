from .logging_setup import initialize_logging
from .tokenizer import generate_text_chunks
from .qa_generation import generate_qa_pairs
from .dataset_processor import create_dataset_from_qa_pairs
from groq_qa_generator.huggingface_api import upload


def generate(config=None):
    """Generate question-answer pairs using user-defined configuration.

    This function allows developers to utilize the pip-installed package by providing their own custom configuration.
    The configuration is used to read input data, process it into chunks, and generate question-answer pairs
    based on those chunks.

    Args:
        config (dict, optional): A dictionary containing user-defined configuration options. The expected
        configuration keys and their descriptions are as follows:

        - system_prompt (str): Path to the system prompt file.
        - sample_question (str): Path to the sample question file.
        - input_data (str): Path to the input data file containing text to process.
        - output_file (str): Path to the output file where generated QA pairs will be saved.
        - model (str): Model name or identifier for QA generation.
        - chunk_size (int): Number of tokens per text chunk (default is 512).
        - tokens_per_question (int): Number of tokens allocated for each question (default is 60).
        - temperature (float): Temperature setting to control randomness in the model's output (default is 0.7).
        - max_tokens (int): Maximum number of tokens in the response (default is 1024).

        Example of usage in `main.py`:
        ```python
        from groq_qa_generator import groq_qa

        # Define custom configuration
        custom_config = {
            "system_prompt": "custom_system_prompt.txt",
            "sample_question": "custom_sample_question.txt",
            "input_data": "custom_input_data.txt",
            "output_file": "custom_qa_output.txt",
            "model": "llama3-70b-8192",
            "chunk_size": 512,
            "tokens_per_question": 60,
            "temperature": 0.1,
            "max_tokens": 1500,
            "split_ratio": 0.7,
            "huggingface_repo": "username/custom-dataset"
        }
        ```
    Returns:
        dict: A dictionary containing the train and test datasets as separate lists of question-answer pairs.
        Example:
        {
            "train": [train_qa_pairs],
            "test": [test_qa_pairs]
        }
    """
    initialize_logging()

    # If config is provided, use it; otherwise, raise an error
    if config is None:
        raise ValueError("config must be provided for generating QA pairs.")

    # Create a config dictionary
    config = {
        "system_prompt": config.get("system_prompt"),
        "sample_question": config.get("sample_question"),
        "input_data": config.get("input_data"),
        "output_file": config.get("output_file"),
        "split_ratio": 0.8,
        "huggingface_repo": "username/dataset",
        "model": config.get("model"),
        "chunk_size": config.get("chunk_size", 512),
        "tokens_per_question": config.get("tokens_per_question", 60),
        "temperature": config.get("temperature", 0.7),
        "max_tokens": config.get("max_tokens", 1024),
    }

    # Read input data and chunk the text
    text_chunks = generate_text_chunks(
        config["input_data"], chunk_size=config.get("chunk_size", 512)
    )

    # Generate QA pairs based on the input text and configuration
    qa_pairs = generate_qa_pairs(text_chunks, config)

    # Retrieve the split ratio from the config
    if "split_ratio" in config:
        split_ratio = config["split_ratio"]

    # Process the qa_pairs and split them into training and test datasets in Hugging Face format
    train_dataset, test_dataset = create_dataset_from_qa_pairs(qa_pairs, split_ratio)

    # Upload the QA pairs to Hugging Face using the repository slug from the config
    if "huggingface_repo" in config:
        upload(train_dataset, test_dataset, config["huggingface_repo"])

    # Return the QA pairs split into training and test datasets
    return {"train": train_dataset, "test": test_dataset}


if __name__ == "__main__":
    raise RuntimeError(
        "Do not run groq_qa.py directly. Please use cli.py for testing and development purposes."
    )
