import pytest
from unittest.mock import patch, mock_open, MagicMock
from datasets import Dataset
from groq_qa_generator.dataset_processor import (
    create_dataset_from_qa_pairs,
    save_datasets,
)


@pytest.fixture
def sample_qa_pairs():
    """
    Fixture providing a sample list of question-answer (QA) pairs for testing.

    These QA pairs are based on Alice in Wonderland text and are used for testing
    various functions that process QA pairs, such as creating datasets and saving them in different formats.

    Returns:
        list of dict: A list of dictionaries containing 'question' and 'answer' key-value pairs.
    """
    return [
        {
            "question": "What did Alice find on the three-legged glass table that gave her hope of escaping the hall?",
            "answer": "A tiny golden key that might unlock one of the doors in the hall.",
        },
        {
            "question": "What was Alice worried about when she thought about her cat Dinah while falling down the hole?",
            "answer": "Alice was worried that Dinah would miss her and hoped someone would remember to give Dinah her saucer of milk at tea-time.",
        },
        {
            "question": "What was Alice's plan when she found the small cake with the words 'EAT ME' on it?",
            "answer": "Alice planned to eat the cake, hoping it would make her grow larger to reach the key or smaller to creep under the door, so she could get into the garden.",
        },
        {
            "question": "What did Alice notice as she fell down the well, and what did she do with a jar she took down from one of the shelves?",
            "answer": "She saw cupboards, book-shelves, maps, and pictures on the sides of the well, and she took down an empty jar labelled 'ORANGE MARMALADE' and put it back into a cupboard as she continued to fall.",
        },
    ]


@patch("groq_qa_generator.dataset_processor.qa_table")
def test_create_dataset_from_qa_pairs(mock_qa_table, sample_qa_pairs):
    """
    Test the `create_dataset_from_qa_pairs` function to verify dataset creation and train/test split.

    This test ensures that the function properly converts a list of QA pairs into Hugging Face Dataset
    objects, applies the specified train/test split (75% train, 25% test in this case), and calls the
    `qa_table` function to log the resulting datasets.

    Args:
        mock_qa_table (MagicMock): Mock for the `qa_table` function used to verify correct calls.
        sample_qa_pairs (list of dict): The sample QA pairs to be converted into datasets.

    Asserts:
        - The train and test datasets are instances of `Dataset`.
        - The correct split ratio is applied (3 training items and 1 test item).
        - `qa_table` is called twice, once for the training dataset and once for the test dataset.
    """
    train_dataset, test_dataset = create_dataset_from_qa_pairs(
        sample_qa_pairs, split_ratio=0.75
    )

    # Assert that the datasets are of the correct type
    assert isinstance(train_dataset, Dataset)
    assert isinstance(test_dataset, Dataset)

    # Ensure that the correct split ratio was applied (3 train and 1 test)
    assert len(train_dataset) == 3
    assert len(test_dataset) == 1

    # Assert that qa_table was called twice (once for train, once for test)
    assert mock_qa_table.call_count == 2
    mock_qa_table.assert_any_call("Training QA Pairs", train_dataset)
    mock_qa_table.assert_any_call("Test QA Pairs", test_dataset)


@patch("builtins.open", new_callable=mock_open)
@patch("json.dump")
def test_save_datasets_json_format(mock_json_dump, mock_file, sample_qa_pairs):
    """
    Test the `save_datasets` function when saving in JSON format.

    This test verifies that the function correctly writes both the training and testing datasets
    as JSON files. It mocks the file handling and ensures that `json.dump` is called to serialize
    the datasets into JSON format.

    Args:
        mock_json_dump (MagicMock): Mock for the `json.dump` function used to verify JSON serialization.
        mock_file (MagicMock): Mock for the `open` function used to verify file opening and writing.
        sample_qa_pairs (list of dict): The sample QA pairs used as mock datasets.

    Asserts:
        - Files for train and test datasets are opened correctly.
        - `json.dump` is called twice (once for the train dataset, once for the test dataset).
    """
    # Mock the train and test datasets
    mock_train_dataset = MagicMock(spec=Dataset)
    mock_test_dataset = MagicMock(spec=Dataset)

    # Mock the dataset's to_dict method to return a list of dictionaries (matching the function's expectations)
    mock_train_dataset.to_dict.return_value = sample_qa_pairs[:2]
    mock_test_dataset.to_dict.return_value = sample_qa_pairs[2:]

    # Call save_datasets with the JSON format enabled
    save_datasets(
        mock_train_dataset, mock_test_dataset, "output.json", json_format=True
    )

    # Assert that files were opened for writing
    assert mock_file.call_count == 2
    mock_file.assert_any_call("output_train.json", "w")
    mock_file.assert_any_call("output_test.json", "w")

    # Assert that json.dump was called twice to write both train and test datasets
    assert mock_json_dump.call_count == 2
    mock_json_dump.assert_any_call(mock_train_dataset.to_dict(), mock_file(), indent=4)
    mock_json_dump.assert_any_call(mock_test_dataset.to_dict(), mock_file(), indent=4)


@patch("builtins.open", new_callable=mock_open)
@patch("logging.info")
def test_save_datasets_text_format(mock_logging_info, mock_file, sample_qa_pairs):
    """
    Test the `save_datasets` function when saving in plain text format.

    This test verifies that the function correctly writes the QA pairs in plain text format,
    ensuring that each question-answer pair is written to the respective train and test text files.

    Args:
        mock_logging_info (MagicMock): Mock for the `logging.info` function used to verify logging.
        mock_file (MagicMock): Mock for the `open` function used to verify file opening and writing.
        sample_qa_pairs (list of dict): The sample QA pairs used as mock datasets.

    Asserts:
        - Files for train and test datasets are opened correctly.
        - The correct question-answer pairs are written to the text files.
        - Logging messages confirm successful writing to the output files.
    """
    # Mock the train and test datasets
    mock_train_dataset = MagicMock(spec=Dataset)
    mock_test_dataset = MagicMock(spec=Dataset)

    # Mock the dataset's to_dict method to return a list of dictionaries (matching the function's expectations)
    mock_train_dataset.to_dict.return_value = sample_qa_pairs[:2]
    mock_test_dataset.to_dict.return_value = sample_qa_pairs[2:]

    # Call save_datasets with plain text format enabled
    save_datasets(
        mock_train_dataset, mock_test_dataset, "output.txt", json_format=False
    )

    # Assert that files were opened for writing
    assert mock_file.call_count == 2
    mock_file.assert_any_call("output_train.txt", "w")
    mock_file.assert_any_call("output_test.txt", "w")

    # Verify that the correct text was written to the train and test files
    mock_file().write.assert_any_call(
        "What did Alice find on the three-legged glass table that gave her hope of escaping the hall?\n"
        "A tiny golden key that might unlock one of the doors in the hall.\n\n"
    )
    mock_file().write.assert_any_call(
        "What was Alice worried about when she thought about her cat Dinah while falling down the hole?\n"
        "Alice was worried that Dinah would miss her and hoped someone would remember to give Dinah her saucer of milk at tea-time.\n\n"
    )

    # Assert that logging.info was called for successful writing
    mock_logging_info.assert_any_call(
        "Train dataset successfully written to output_train.txt"
    )
    mock_logging_info.assert_any_call(
        "Test dataset successfully written to output_test.txt"
    )


@patch("builtins.open", new_callable=mock_open)
@patch("json.dump")
def test_save_datasets_handles_empty_dataset(mock_json_dump, mock_file):
    """
    Test the `save_datasets` function with empty datasets.

    This test verifies that when the datasets are empty, the function handles this case gracefully
    and still writes empty files without errors.

    Args:
        mock_json_dump (MagicMock): Mock for the `json.dump` function used to verify JSON serialization.
        mock_file (MagicMock): Mock for the `open` function used to verify file opening and writing.

    Asserts:
        - Files for train and test datasets are opened even if the datasets are empty.
        - `json.dump` is called for both the train and test datasets (even if they are empty).
    """
    # Mock empty datasets
    mock_train_dataset = MagicMock(spec=Dataset)
    mock_test_dataset = MagicMock(spec=Dataset)

    # Mock the dataset's to_dict method to return an empty list (matching the function's expectations)
    mock_train_dataset.to_dict.return_value = []
    mock_test_dataset.to_dict.return_value = []

    # Call save_datasets with the JSON format enabled
    save_datasets(
        mock_train_dataset, mock_test_dataset, "output.json", json_format=True
    )
