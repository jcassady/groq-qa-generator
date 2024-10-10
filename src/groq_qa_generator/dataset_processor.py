import json
import logging
from .logging_setup import qa_table
from datasets import Dataset


def create_dataset_from_qa_pairs(qa_pairs, split_ratio=0.8):
    """
    Converts the qa_pairs into a Hugging Face dataset and performs a train/test split.

    Args:
        qa_pairs (list of dict): List of QA pairs.
        split_ratio (float): Percentage of training data (e.g., 0.8 for 80% train, 20% test).

    Returns:
        tuple: Two Hugging Face datasets - (train_dataset, test_dataset).
    """
    # Calculate the test size (which is 1 - split_ratio)
    test_size = 1 - split_ratio

    # Convert the qa_pairs into a Hugging Face dataset
    dataset = Dataset.from_dict(
        {
            "question": [pair["question"] for pair in qa_pairs],
            "answer": [pair["answer"] for pair in qa_pairs],
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


def save_datasets(train_dataset, test_dataset, output_file, json_format=False):
    # Convert Dataset objects to lists of dictionaries for JSON serialization
    train_data_list = train_dataset.to_dict()
    test_data_list = test_dataset.to_dict()

    # Determine the output filenames for train/test datasets with .json extension
    if json_format:
        train_file = output_file.replace(".json", "_train.json")
        test_file = output_file.replace(".json", "_test.json")
    else:
        train_file = output_file.replace(".txt", "_train.txt")
        test_file = output_file.replace(".txt", "_test.txt")

    # Check if the json_format argument is set
    if json_format:
        # If saving in JSON format, dump the datasets as JSON objects
        with open(train_file, "w") as train_f:
            json.dump(train_data_list, train_f, indent=4)
        logging.info(f"Train dataset successfully written to {train_file}")
        with open(test_file, "w") as test_f:
            json.dump(test_data_list, test_f, indent=4)
        logging.info(f"Test dataset successfully written to {test_file}")
    else:
        # If saving in plain text format, write each question-answer pair as text
        with open(train_file, "w") as train_f:
            for pair in train_data_list:
                train_f.write(f"{pair['question']}\n{pair['answer']}\n\n")
        logging.info(f"Train dataset successfully written to {train_file}")
        with open(test_file, "w") as test_f:
            for pair in test_data_list:
                test_f.write(f"{pair['question']}\n{pair['answer']}\n\n")
        logging.info(f"Test dataset successfully written to {test_file}")
