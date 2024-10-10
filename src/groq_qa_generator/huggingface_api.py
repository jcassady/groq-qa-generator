import logging
from datasets import DatasetDict


def upload(train_dataset, test_dataset, repo):
    """
    Upload the training and test QA pairs to Hugging Face Dataset Hub
    using DatasetDict.

    Args:
        train_dataset (Dataset): Hugging Face dataset for training.
        test_dataset (Dataset): Hugging Face dataset for testing.
        repo (str): The repository on Hugging Face.

    Returns:
        None
    """
    # Create DatasetDict containing both train and eval datasets
    train_test_dataset = DatasetDict({"train": train_dataset, "eval": test_dataset})

    # Push the dataset to Hugging Face Hub using the provided repo
    if repo:
        logging.info("Uploading QA dataset to Hugging Face Hub.")
        train_test_dataset.push_to_hub(repo, private=True)
        logging.info(
            f"Dataset uploaded to Hugging Face hub at https://huggingface.co/datasets/{repo}"
        )
    else:
        logging.warning(
            "Hugging Face Hub repository not provided. Skipping QA dataset upload."
        )
