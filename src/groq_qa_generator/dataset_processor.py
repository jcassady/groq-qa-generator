import json
import logging
from .logging_setup import qa_table
from datasets import Dataset


def create_dataset_from_qa_pairs(qa_pairs, split_ratio=0.8):
    """
    Converts a list of question-answer pairs into a Hugging Face Dataset and performs a train/test split.

    Args:
        qa_pairs (list of dict): A list of dictionaries, each containing 'question' and 'answer' keys.
        split_ratio (float, optional): The proportion of the data to use for training.

    Returns:
        tuple: A tuple containing the training dataset and testing dataset as Hugging Face Dataset objects.
    """
    # Filter out malformed pairs (those missing question or answer)
    valid_qa_pairs = [pair for pair in qa_pairs if pair["question"] and pair["answer"]]

    # Calculate the test size (which is 1 - split_ratio)
    test_size = 1 - split_ratio

    # Convert the valid qa_pairs into a Hugging Face dataset
    dataset = Dataset.from_dict(
        {
            "question": [pair["question"] for pair in valid_qa_pairs],
            "answer": [pair["answer"] for pair in valid_qa_pairs],
        }
    )

    # Perform the train/test split
    train_test_data = dataset.train_test_split(test_size=test_size)

    # Extract the train and test datasets
    train_dataset = train_test_data["train"]
    test_dataset = train_test_data["test"]

    # Display the QA table for the training and test datasets
    qa_table("Training QA Pairs", train_dataset)
    qa_table("Test QA Pairs", test_dataset)

    return train_dataset, test_dataset


def save_datasets(train_dataset, test_dataset, output_file, json_format=True):
    """
    Saves the train and test datasets to the specified output files in either JSON or plain text format.
    
    Args:
        train_dataset (Dataset): Hugging Face dataset containing the training data.
        test_dataset (Dataset): Hugging Face dataset containing the testing data.
        output_file (str): The base path for the output file. The function will append '_train' and '_test'
                           to the base path for the respective datasets.
        json_format (bool): If True, saves the datasets in JSON format. If False, saves in plain text format.
    
    This function handles both saving formats. In JSON format, it saves the datasets as structured JSON files.
    In plain text format, it saves the question-answer pairs line by line, skipping any malformed entries.
    
    After saving the datasets, the paths to the train and test files are logged using `logging.info`.
    
    Example:
        save_datasets(train_dataset, test_dataset, 'output.txt', json_format=False)
        This would save the datasets as 'output_train.txt' and 'output_test.txt'.
    """
    # Determine file paths for train and test sets
    train_file, test_file = get_output_file_paths(output_file, json_format)

    if json_format:
        # Save as JSON format
        save_as_json(train_dataset, train_file)
        save_as_json(test_dataset, test_file)
        # Log the paths to the output JSON files
        logging.info(f"Train dataset written to {train_file}")
        logging.info(f"Test dataset written to {test_file}")
    else:
        # Save as plain text format
        train_data = train_dataset.to_dict()
        test_data = test_dataset.to_dict()

        # Write the text data
        save_as_text(train_data, train_file)
        save_as_text(test_data, test_file)

        # Log the paths to the output text files
        logging.info(f"Train dataset written to {train_file}")
        logging.info(f"Test dataset written to {test_file}")



def get_output_file_paths(output_file, json_format):
    """
    Generates the output file paths for the training and testing datasets based on the file format.

    Args:
        output_file (str): The base output file path (e.g., 'output.txt' or 'output.json').
        json_format (bool): Whether to generate file paths for JSON files (True) or plain text files (False).

    Returns:
        tuple: A tuple containing the file paths for the training and testing datasets.
    """
    if json_format:
        train_file = output_file.replace(".json", "_train.json")
        test_file = output_file.replace(".json", "_test.json")
    else:
        train_file = output_file.replace(".txt", "_train.txt")
        test_file = output_file.replace(".txt", "_test.txt")
    return train_file, test_file


def save_as_text(data, file_path):
    """
    Save the dataset to a text file with each question-answer pair separated by newlines.
    """
    with open(file_path, "w") as f:
        for question, answer in zip(data["question"], data["answer"]):
            # Ensure valid data before writing
            if question and answer:
                f.write(f"{question}\n{answer}\n\n")
            else:
                print(
                    f"Skipping entry with missing question or answer: {question}, {answer}"
                )


def save_as_json(data, file_path):
    """
    Saves a dataset in JSON format with proper indentation.

    Args:
        data (Dataset): The dataset to save in JSON format.
        file_path (str): The path where the dataset will be saved in JSON format.

    Returns:
        None
    """
    with open(file_path, "w") as f:
        json.dump(data.to_dict(), f, indent=4)


def save_text_data(data, file_path, dataset_type):
    """
    Saves a list of question-answer pairs in plain text format, ensuring each pair is valid.
    Invalid pairs (missing question or answer) are skipped, and a warning is logged.

    Args:
        data (list of dict): A list of dictionaries containing 'question' and 'answer' keys.
        file_path (str): The file path where the dataset will be saved in text format.
        dataset_type (str): A string indicating the dataset type ('train' or 'test'), used for logging purposes.

    Returns:
        None
    """
    with open(file_path, "w") as f:
        for index, pair in enumerate(data):
            if "question" in pair and "answer" in pair:
                f.write(f"{pair['question']}\n{pair['answer']}\n\n")
            else:
                logging.warning(
                    f"Skipping malformed pair #{index + 1} in {dataset_type} dataset: {pair}"
                )
    logging.info(
        f"{dataset_type.capitalize()} dataset successfully written to {file_path}"
    )
