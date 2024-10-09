from .logging_setup import initialize_logging
from .tokenizer import generate_text_chunks
from .qa_generation import generate_qa_pairs


def generate(custom_config=None):
    """Generate question-answer pairs using user-defined configuration.

    This function allows developers to utilize the pip-installed package by providing their own custom configuration.
    The configuration is used to read input data, process it into chunks, and generate question-answer pairs
    based on those chunks.

    Args:
        custom_config (dict, optional): A dictionary containing user-defined configuration options. The expected
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
            "max_tokens": 1500
        }

        # Generate question-answer pairs
        qa_pairs = groq_qa.generate(custom_config)

        # Print generated QA pairs
        for pair in qa_pairs:
            print(pair)
        ```

    Returns:
        list of dict: A list of dictionaries containing generated question-answer pairs.
    """
    initialize_logging()

    # If custom_config is provided, use it; otherwise, raise an error
    if custom_config is None:
        raise ValueError("custom_config must be provided for generating QA pairs.")

    # Create a default groq_config based on the custom_config
    groq_config = {
        "system_prompt": custom_config.get("system_prompt"),
        "sample_question": custom_config.get("sample_question"),
        "input_data": custom_config.get("input_data"),
        "output_file": custom_config.get("output_file"),
        "model": custom_config.get("model"),
        "chunk_size": custom_config.get("chunk_size", 512),
        "tokens_per_question": custom_config.get("tokens_per_question", 60),
        "temperature": custom_config.get("temperature", 0.7),
        "max_tokens": custom_config.get("max_tokens", 1024),
    }

    # Read input data and chunk the text
    text_chunks = generate_text_chunks(
        groq_config["input_data"], chunk_size=groq_config.get("chunk_size", 512)
    )

    # Generate QA pairs based on the input text and configuration
    qa_pairs = generate_qa_pairs(text_chunks, groq_config)

    return qa_pairs
