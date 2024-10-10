from groq_qa_generator.logging_setup import initialize_logging
from groq_qa_generator.config import (
    parse_arguments,
    initialize_user_config,
    load_config,
)
from groq_qa_generator.tokenizer import generate_text_chunks
from groq_qa_generator.qa_generation import generate_qa_pairs
from groq_qa_generator.dataset_processor import (
    create_dataset_from_qa_pairs,
    save_datasets,
)
from groq_qa_generator.huggingface_api import upload


def main():
    """Main entry point for the QA pair generation process.

    This function orchestrates the entire process of parsing command-line arguments,
    setting up logging, initializing the user configuration directory, loading the
    configuration, reading and chunking the input text, and generating question-answer pairs.
    """

    args = parse_arguments()

    initialize_logging()  # Initialize logging for the application

    initialize_user_config()  # Set up user configuration files

    config = load_config(args)  # Load configuration settings from the config file

    text_chunks = generate_text_chunks(
        config["input_data"], chunk_size=config.get("chunk_size", 512)
    )  # Read input data and chunk it into manageable pieces

    qa_pairs = generate_qa_pairs(
        text_chunks, config
    )  # Generate question-answer pairs from text chunks

    # Retrieve the split ratio from the config
    if "split_ratio" in config:
        split_ratio = config["split_ratio"]

    # Process the qa_pairs and split them into training and test datasets in Hugging Face format
    train_dataset, test_dataset = create_dataset_from_qa_pairs(qa_pairs, split_ratio)

    # Save the datasets to local storage
    save_datasets(
        train_dataset, test_dataset, config["output_file"], json_format=args.json
    )

    # Upload the QA pairs to Hugging Face using the repository slug from the config
    if "huggingface_repo" in config:
        upload(train_dataset, test_dataset, config["huggingface_repo"])


if __name__ == "__main__":
    main()  # Run the main function if the script is executed directly
