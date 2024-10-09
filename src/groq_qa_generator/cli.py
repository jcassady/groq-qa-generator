from groq_qa_generator.logging_setup import initialize_logging
from groq_qa_generator.config import (
    parse_arguments,
    initialize_user_config,
    load_config,
)
from groq_qa_generator.tokenizer import generate_text_chunks
from groq_qa_generator.qa_generation import generate_qa_pairs


def main():
    """Main entry point for the QA pair generation process.

    This function orchestrates the entire process of parsing command-line arguments,
    setting up logging, initializing the user configuration directory, loading the 
    configuration, reading and chunking the input text, and generating question-answer pairs.
    """

    args = parse_arguments()

    initialize_logging()  # Initialize logging for the application.

    initialize_user_config()  # Set up user configuration files.

    groq_config = load_config(args)  # Load configuration settings from the config file.

    text_chunks = generate_text_chunks(
        groq_config["input_data"], chunk_size=groq_config.get("chunk_size", 512)
    )  # Read input data and chunk it into manageable pieces.

    qa_pairs = generate_qa_pairs(
        text_chunks, groq_config
    )  # Generate question-answer pairs from text chunks.


if __name__ == "__main__":
    main()  # Run the main function if the script is executed directly.
