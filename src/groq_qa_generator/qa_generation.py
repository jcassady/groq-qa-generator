import logging
from .logging_setup import initialize_logging
from .groq_api import (
    get_api_key,
    get_groq_client,
    get_groq_completion,
    stream_completion,
)
from .text_processing import write_response_to_file

# from .config import args


def load_sample_question(file_path):
    """Load the sample question from a file.

    Args:
        file_path (str): The path to the file containing the sample question.

    Returns:
        str: The content of the sample question file, stripped of leading/trailing whitespace.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_system_prompt(file_path, chunk_size, tokens_per_question, questions=None):
    """
    Load and prepare the system prompt, adjusting it based on chunk size, tokens per question, or a user-specified number of questions.

    This function reads the system prompt from a file and determines how many questions should
    be generated. If the user provides a specific number of questions via the `questions` argument,
    that number is used. Otherwise, the number of questions is calculated based on the chunk size
    and tokens per question. The calculated or provided number of questions then replaces a placeholder
    ("{questions_per_chunk}") in the system prompt.

    Args:
        file_path (str): The path to the system prompt file.
        chunk_size (int): The number of tokens per chunk (default: 512).
        tokens_per_question (int): The number of tokens allocated for each question (default: 60).
        questions (int, optional): The number of questions specified by the user, typically passed
                                   through the CLI `--questions` argument. If provided, this overrides
                                   the calculation based on chunk size and tokens per question.

    Returns:
        str: The formatted system prompt with the number of questions inserted.

    Notes:
        - With a default `chunk_size` of 512 and `tokens_per_question` of 60, this function
          will generate approximately 8 questions per chunk (512 / 60 ≈ 8).
        - If the `--questions` CLI argument is provided, its value is used directly instead
          of calculating the number of questions from the chunk size and tokens per question.
        - If `questions` is not provided, the number of questions is dynamically calculated as:
          questions_per_chunk = chunk_size // tokens_per_question.
    """

    # Use the provided number of questions, or calculate based on chunk_size and tokens_per_question
    if questions is not None:
        questions_per_chunk = questions
    else:
        questions_per_chunk = int(chunk_size / tokens_per_question)

    with open(file_path, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    # Replace the placeholder for questions with the calculated or provided value
    modified_prompt = system_prompt.replace(
        "<n>", str(questions_per_chunk)
    )

    return modified_prompt


def create_groq_prompt(system_prompt, sample_question):
    """Combine the system prompt and sample question into a full prompt for Groq API.

    Args:
        system_prompt (str): The system prompt defining the behavior and tone of the model.
        sample_question (str): A sample question to help guide the generation.

    Returns:
        str: The full combined prompt including both the system prompt and sample question.
    """
    full_prompt = f"{system_prompt}\n\n{sample_question}"

    return full_prompt


def generate_qa_pairs(text_chunks, groq_config):
    """
    Generate question-answer pairs from text chunks using the Groq API.

    This function processes a list of text chunks and generates question-answer (QA) pairs based on
    the provided configuration. It leverages the Groq API for completion and logs the output to an
    output file, either in JSON or plain text format.

    Args:
        text_chunks (list of str): The list of text chunks to generate questions from.
        groq_config (dict): Configuration settings containing:
            - api_key (str): API key for accessing Groq.
            - system_prompt (str): Path to the system prompt file.
            - sample_question (str): Path to the sample question file.
            - chunk_size (int): Number of tokens per text chunk.
            - tokens_per_question (int): Number of tokens allocated for each question.
            - model (str): Model name or identifier for QA generation.
            - temperature (float): Temperature setting to control randomness in the model's output.
            - max_tokens (int): Maximum number of tokens in the response.
            - questions (int, optional): The number of questions to generate, overriding chunk size calculation.
            - output_file (str): Path to the output file where results will be written.

    Returns:
        list of dict: A list of dictionaries containing generated question-answer pairs.

    Logs:
        Logs success and failure for each text chunk processed.
    """

    def process_chunk(
        client,
        chunk_index,
        total_chunks,
        chunk_text,
        system_prompt,
        sample_question,
        groq_config,
        all_qa_pairs,
    ):
        """
        Process a single chunk of text and generate QA pairs.

        This function handles the processing of a single text chunk, generating QA pairs based on
        the system prompt and sample question. It logs the progress and captures any errors encountered
        during processing.

        Args:
            client (object): The Groq client used for generating completions.
            chunk_index (int): The index of the current text chunk being processed.
            total_chunks (int): The total number of text chunks to be processed.
            chunk_text (str): The text chunk to process.
            system_prompt (str): The system prompt to be used for generating QA pairs.
            sample_question (str): The sample question to be used for generating QA pairs.
            groq_config (dict): The configuration settings including model, temperature, and max tokens.
            all_qa_pairs (list): A list to store all generated QA pairs.

        Side Effects:
            Updates the `all_qa_pairs` list with new QA pairs from the current chunk.
        """
        logging.info(f"Processing chunk {chunk_index + 1} of {total_chunks}")

        full_prompt_text = create_groq_prompt(system_prompt, sample_question)
        completion = get_groq_completion(
            client,
            full_prompt_text,
            chunk_text,
            groq_config["model"],
            groq_config["temperature"],
            groq_config["max_tokens"],
        )

        if completion:
            response = stream_completion(completion)
            logging.info(response)
            qa_pairs = parse_qa_pairs(response)
            all_qa_pairs.extend(qa_pairs)
        else:
            logging.error(f"Failed to generate QA pairs for chunk {chunk_index}.")

    def parse_qa_pairs(response):
        """
        Parse the model's response into question-answer pairs.

        This function splits the response from the model into individual question-answer
        pairs based on double newline characters.

        Args:
            response (str): The raw response text generated by the model.

        Returns:
            list of str: A list of question-answer pairs.
        """
        return response.strip().split("\n\n")

    def write_qa_pairs_to_file(qa_pairs, output_file, as_json):
        """
        Write the QA pairs to the specified output file, in JSON format if required.

        This function writes the accumulated QA pairs to the output file. If the `as_json`
        flag is set, the output will be written in JSON format; otherwise, it will be plain text.

        Args:
            qa_pairs (list of str): The list of question-answer pairs to be written to the file.
            output_file (str): The path to the output file where the QA pairs will be saved.
            as_json (bool): Whether to save the QA pairs in JSON format.

        Side Effects:
            Writes the QA pairs to the specified output file.
        """
        content = "\n\n".join(qa_pairs)
        write_response_to_file(content, output_file, as_json)

    api_key = get_api_key()
    client = get_groq_client(api_key)

    system_prompt = load_system_prompt(
        groq_config["system_prompt"],
        groq_config["chunk_size"],
        groq_config["tokens_per_question"],
        groq_config["questions"],
    )

    sample_question = load_sample_question(groq_config["sample_question"])

    all_qa_pairs = []
    total_chunks = len(text_chunks)

    for chunk_index, chunk_text in enumerate(text_chunks):
        process_chunk(
            client,
            chunk_index,
            total_chunks,
            chunk_text,
            system_prompt,
            sample_question,
            groq_config,
            all_qa_pairs,
        )

    write_qa_pairs_to_file(
        all_qa_pairs, groq_config["output_file"], groq_config["json"]
    )

    logging.info(f"QA pairs written to file: {groq_config['output_file']}")

    return all_qa_pairs
