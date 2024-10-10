import textwrap
import logging
from .logging_setup import initialize_logging
from .groq_api import (
    get_api_key,
    get_groq_client,
    get_groq_completion,
    stream_completion,
)


def load_sample_question(file_path):
    """Load the sample question from a file.

    Args:
        file_path (str): The path to the file containing the sample question.

    Returns:
        str: The content of the sample question file, stripped of leading/trailing whitespace.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            sample_question = f.read().strip()
            logging.info(f"Sample question loaded from {file_path}.")
            return sample_question
    except Exception as e:
        logging.error(f"Failed to load sample question from {file_path}: {e}")
        return ""


def load_system_prompt(file_path, chunk_size, tokens_per_question, questions=None):
    """
    Load and prepare the system prompt, adjusting it based on chunk size, tokens per question,
    or a user-specified number of questions.

    Args:
        file_path (str): The path to the system prompt file.
        chunk_size (int): The number of tokens per chunk (default: 512).
        tokens_per_question (int): The number of tokens allocated for each question (default: 60).
        questions (int, optional): The number of questions specified by the user, typically passed
                                   through the CLI `--questions` argument. If provided, this overrides
                                   the calculation based on chunk size and tokens per question.

    Returns:
        str: The formatted system prompt with the number of questions inserted.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
            logging.info(f"System prompt loaded from {file_path}.")
    except Exception as e:
        logging.error(f"Failed to load system prompt from {file_path}: {e}")
        system_prompt = ""

    if questions is not None:
        questions_per_chunk = questions
    else:
        questions_per_chunk = int(chunk_size / tokens_per_question)

    # Replace the placeholder for questions with the calculated or provided value
    modified_prompt = system_prompt.replace("<n>", str(questions_per_chunk))
    logging.info(
        f"System prompt configured with {questions_per_chunk} questions per chunk."
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
    logging.debug("Groq prompt created by combining system prompt and sample question.")
    return full_prompt


def generate_qa_pairs(text_chunks, groq_config):
    """
    Generate question-answer pairs from text chunks using the Groq API.

    This function processes a list of text chunks and generates question-answer (QA) pairs based on
    the provided configuration. It leverages the Groq API for completion and logs the output to an
    output file in JSON or plain text format.

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
            - json (bool): Flag indicating whether to write output in JSON format.

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
            # Stream the full response and log it for debugging
            response = stream_completion(completion)
            logging.debug(f"Raw response for chunk {chunk_index + 1}: {response}")

            # Parse the response into QA pairs and add them to the full list
            qa_pairs = parse_qa_pairs(response)
            all_qa_pairs.extend(qa_pairs)

            # Function to wrap text and pad it to the right to match the box width
            def wrap_and_format_text(text, width):
                wrapped_lines = textwrap.wrap(text, width=width)
                return [f"| {line:<{width}} |" for line in wrapped_lines]

            # Log each generated QA pair to the console inside an ASCII box
            for index, qa_pair in enumerate(qa_pairs, start=1):
                question = qa_pair["question"]
                answer = qa_pair["answer"]

                # Wrap and format the question and answer with a 100 character limit
                wrapped_question = wrap_and_format_text(f"Q: {question}", 100)
                wrapped_answer = wrap_and_format_text(f"A: {answer}", 100)

                # Determine the width of the box (100 characters + 2 for borders)
                box_width = 100 + 2

                # Create the top and bottom border of the box
                top_border = f"+{'-' * box_width}+"
                separator = f"| {'-' * 100} |"

                # Log the formatted QA pair in an ASCII box
                logging.info(f"Question #{index}:")
                logging.info(top_border)
                for line in wrapped_question:
                    logging.info(line)
                logging.info(separator)
                for line in wrapped_answer:
                    logging.info(line)
                logging.info(top_border)
                logging.info("\n")  # Add a newline for spacing between QA pairs

            logging.info(
                f"Generated {len(qa_pairs)} QA pairs for chunk {chunk_index + 1}."
            )
        else:
            logging.error(f"Failed to generate QA pairs for chunk {chunk_index + 1}.")

    def parse_qa_pairs(response):
        """
        Parse the model's response into question-answer pairs.

        This function splits the response from the model into individual question-answer
        pairs based on double newline characters and formats them into dictionaries.

        Args:
            response (str): The raw response text generated by the model.

        Returns:
            list of dict: A list of dictionaries with 'question' and 'answer' keys.
        """
        qa_list = []
        # Split the response by double newlines to separate Q&A pairs
        raw_pairs = response.strip().split("\n\n")
        for raw_pair in raw_pairs:
            # Further split each pair by single newline to separate question and answer
            parts = raw_pair.split("\n", 1)
            if len(parts) == 2:
                question, answer = parts
                qa_dict = {"question": question.strip(), "answer": answer.strip()}
                qa_list.append(qa_dict)
            else:
                logging.warning(f"Unexpected Q&A format: {raw_pair}")
        return qa_list

    # Main function execution
    api_key = get_api_key()
    client = get_groq_client(api_key)

    system_prompt = load_system_prompt(
        groq_config["system_prompt"],
        groq_config["chunk_size"],
        groq_config["tokens_per_question"],
        groq_config["questions"],
    )

    sample_question = load_sample_question(groq_config["sample_question"])

    # Append to qa_pairs list for each iteration on text_chunks
    qa_pairs = []
    total_chunks = len(text_chunks)

    # Process each text chunk, collecting QA pairs
    for chunk_index, chunk_text in enumerate(text_chunks):
        process_chunk(
            client,
            chunk_index,
            total_chunks,
            chunk_text,
            system_prompt,
            sample_question,
            groq_config,
            qa_pairs,
        )

    logging.info(f"Total QA pairs generated: {len(qa_pairs)}")

    return qa_pairs
