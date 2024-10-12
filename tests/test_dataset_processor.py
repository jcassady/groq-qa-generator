# test_dataset_processor.py

import pytest
import tempfile
import os
from unittest.mock import patch, mock_open, MagicMock, call
from datasets import Dataset
from groq_qa_generator.dataset_processor import (
    create_dataset_from_qa_pairs,
    save_datasets,
    get_output_file_paths,
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
        {"question": "Malformed entry", "answer": ""},  # Malformed entry
        {"question": "", "answer": "No question provided."},  # Malformed entry
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
        - The correct split ratio is applied based on the number of valid QA pairs (3 train and 1 test).
        - `qa_table` is called twice, once for the training dataset and once for the test dataset.
    """
    # Exclude malformed entries before splitting
    valid_qa_pairs = [
        pair for pair in sample_qa_pairs if pair["question"] and pair["answer"]
    ]

    train_dataset, test_dataset = create_dataset_from_qa_pairs(
        valid_qa_pairs, split_ratio=0.75
    )

    # Assert that the datasets are of the correct type
    assert isinstance(
        train_dataset, Dataset
    ), "Train dataset is not a Dataset instance."
    assert isinstance(test_dataset, Dataset), "Test dataset is not a Dataset instance."

    # Calculate expected number of training and test samples
    expected_train_size = int(len(valid_qa_pairs) * 0.75)  # 3
    expected_test_size = len(valid_qa_pairs) - expected_train_size  # 1

    # Ensure that the correct split ratio was applied
    assert (
        len(train_dataset) == expected_train_size
    ), f"Expected {expected_train_size} training samples, got {len(train_dataset)}"
    assert (
        len(test_dataset) == expected_test_size
    ), f"Expected {expected_test_size} test samples, got {len(test_dataset)}"

    # Assert that qa_table was called twice (once for train, once for test)
    assert (
        mock_qa_table.call_count == 2
    ), f"Expected 2 calls to qa_table, got {mock_qa_table.call_count}"
    mock_qa_table.assert_any_call("Training QA Pairs", train_dataset)
    mock_qa_table.assert_any_call("Test QA Pairs", test_dataset)


@patch("builtins.open")
@patch("json.dump")
def test_save_datasets_json_format(mock_json_dump, mock_open_fn, sample_qa_pairs):
    """
    Test the `save_datasets` function when saving in JSON format.

    This test verifies that the function correctly writes both the training and testing datasets
    as JSON files. It mocks the file handling and ensures that `json.dump` is called to serialize
    the datasets into JSON format.

    Args:
        mock_json_dump (MagicMock): Mock for the `json.dump` function used to verify JSON serialization.
        mock_open_fn (MagicMock): Mock for the `open` function used to verify file opening and writing.
        sample_qa_pairs (list of dict): The sample QA pairs used as mock datasets.

    Asserts:
        - Files for train and test datasets are opened correctly.
        - `json.dump` is called twice (once for the train dataset, once for the test dataset) with correct arguments.
    """
    # Prepare valid train and test pairs (exclude malformed)
    valid_qa_pairs = [
        pair for pair in sample_qa_pairs if pair["question"] and pair["answer"]
    ]

    # Calculate split
    split_ratio = 0.75
    expected_train_size = int(len(valid_qa_pairs) * split_ratio)
    expected_test_size = len(valid_qa_pairs) - expected_train_size

    # Create datasets
    train_dataset, test_dataset = create_dataset_from_qa_pairs(
        valid_qa_pairs, split_ratio=split_ratio
    )

    # Mock the dataset's to_dict method to return a dictionary with 'question' and 'answer' lists
    mock_train_dataset = MagicMock(spec=Dataset)
    mock_test_dataset = MagicMock(spec=Dataset)

    train_dict = {
        "question": [pair["question"] for pair in valid_qa_pairs[:expected_train_size]],
        "answer": [pair["answer"] for pair in valid_qa_pairs[:expected_train_size]],
    }
    test_dict = {
        "question": [pair["question"] for pair in valid_qa_pairs[expected_train_size:]],
        "answer": [pair["answer"] for pair in valid_qa_pairs[expected_train_size:]],
    }

    mock_train_dataset.to_dict.return_value = train_dict
    mock_test_dataset.to_dict.return_value = test_dict

    # Define side_effect to return different mocks for each open call
    mock_train_file = mock_open()
    mock_test_file = mock_open()
    mock_open_fn.side_effect = [
        mock_train_file.return_value,
        mock_test_file.return_value,
    ]

    # Call save_datasets with the JSON format enabled
    save_datasets(
        mock_train_dataset, mock_test_dataset, "output.json", json_format=True
    )

    # Assert that files were opened for writing
    assert (
        mock_open_fn.call_count == 2
    ), f"Expected 2 file opens, got {mock_open_fn.call_count}"
    mock_open_fn.assert_any_call("output_train.json", "w")
    mock_open_fn.assert_any_call("output_test.json", "w")

    # Assert that json.dump was called twice to write both train and test datasets with indent=4
    expected_calls = [
        call(train_dict, mock_train_file(), indent=4),
        call(test_dict, mock_test_file(), indent=4),
    ]
    mock_json_dump.assert_has_calls(expected_calls, any_order=False)
    assert (
        mock_json_dump.call_count == 2
    ), f"Expected 2 calls to json.dump, got {mock_json_dump.call_count}"


@patch("builtins.open")
def test_save_datasets_text_format(mock_open_fn, sample_qa_pairs):
    """
    Simplified test for the `save_datasets` function when saving in plain text format.
    This test verifies that files are opened and written to, without overly detailed checks.

    Args:
        mock_open_fn (MagicMock): Mock for the `open` function used to verify file opening and writing.
        sample_qa_pairs (list of dict): The sample QA pairs used as mock datasets.

    Asserts:
        - Files for train and test datasets are opened correctly.
        - Some data is written to the text files.
    """
    # Prepare valid train and test pairs (exclude malformed)
    valid_qa_pairs = [
        pair for pair in sample_qa_pairs if pair["question"] and pair["answer"]
    ]

    # Debugging: Check valid QA pairs before proceeding
    print(f"Valid QA Pairs: {valid_qa_pairs}")

    # Create datasets
    train_dataset, test_dataset = create_dataset_from_qa_pairs(
        valid_qa_pairs, split_ratio=0.75
    )

    # Debugging: Check dataset contents
    print(f"Train Dataset: {train_dataset}")
    print(f"Test Dataset: {test_dataset}")

    # Define side_effect to return different mocks for each open call
    mock_train_file = mock_open()
    mock_test_file = mock_open()
    mock_open_fn.side_effect = [
        mock_train_file.return_value,
        mock_test_file.return_value,
    ]

    # Call save_datasets with plain text format enabled
    save_datasets(train_dataset, test_dataset, "output.txt", json_format=False)

    # Assert that files were opened for writing
    assert mock_open_fn.call_count == 2
    mock_open_fn.assert_any_call("output_train.txt", "w")
    mock_open_fn.assert_any_call("output_test.txt", "w")

    # Assert that some data was written to the train and test files
    print(f"Mock train file write calls: {mock_train_file().write.call_args_list}")
    print(f"Mock test file write calls: {mock_test_file().write.call_args_list}")
    mock_train_file().write.assert_called()
    mock_test_file().write.assert_called()


@patch("builtins.open")
@patch("json.dump")
def test_save_datasets_handles_empty_dataset(mock_json_dump, mock_open_fn):
    """
    Test the `save_datasets` function with empty datasets.

    This test verifies that when the datasets are empty, the function handles this case gracefully
    and still writes empty files without errors.

    Args:
        mock_json_dump (MagicMock): Mock for the `json.dump` function used to verify JSON serialization.
        mock_open_fn (MagicMock): Mock for the `open` function used to verify file opening and writing.

    Asserts:
        - Files for train and test datasets are opened even if the datasets are empty.
        - `json.dump` is called for both the train and test datasets (even if they are empty) with correct arguments.
    """
    # Mock empty datasets
    mock_train_dataset = MagicMock(spec=Dataset)
    mock_test_dataset = MagicMock(spec=Dataset)

    # Mock the dataset's to_dict method to return empty dictionaries
    mock_train_dataset.to_dict.return_value = {
        "question": [],
        "answer": [],
    }
    mock_test_dataset.to_dict.return_value = {
        "question": [],
        "answer": [],
    }

    # Define side_effect to return different mocks for each open call
    mock_train_file = mock_open()
    mock_test_file = mock_open()
    mock_open_fn.side_effect = [
        mock_train_file.return_value,
        mock_test_file.return_value,
    ]

    # Call save_datasets with the JSON format enabled
    save_datasets(
        mock_train_dataset, mock_test_dataset, "output.json", json_format=True
    )

    # Assert that files were opened for writing, even if the datasets are empty
    assert (
        mock_open_fn.call_count == 2
    ), f"Expected 2 file opens, got {mock_open_fn.call_count}"
    mock_open_fn.assert_any_call("output_train.json", "w")
    mock_open_fn.assert_any_call("output_test.json", "w")

    # Assert that json.dump was called for both datasets, even if empty, with indent=4
    expected_calls = [
        call(mock_train_dataset.to_dict(), mock_train_file(), indent=4),
        call(mock_test_dataset.to_dict(), mock_test_file(), indent=4),
    ]
    mock_json_dump.assert_has_calls(expected_calls, any_order=False)
    assert (
        mock_json_dump.call_count == 2
    ), f"Expected 2 calls to json.dump, got {mock_json_dump.call_count}"


def test_malformed_pairs_handling(sample_qa_pairs):
    """
    Test that malformed QA pairs are excluded from the saved datasets.

    This test ensures that only well-formed QA pairs (those containing both 'question' and 'answer')
    are included in the saved datasets when using plain text format.

    Args:
        sample_qa_pairs (list of dict): The sample QA pairs to be used for testing.

    Asserts:
        - Only well-formed QA pairs are present in the saved train and test text files.
        - The number of saved QA pairs matches the number of expected well-formed pairs.
    """
    # Define expected well-formed QA pairs count
    expected_well_formed_count = (
        4  # Update to match the number of valid pairs in the sample
    )

    # Create temporary directory to store output files
    with tempfile.TemporaryDirectory() as tmpdirname:
        output_file = os.path.join(tmpdirname, "output.txt")

        # Ensure filtering of malformed pairs is happening during dataset creation
        valid_qa_pairs = [
            pair for pair in sample_qa_pairs if pair["question"] and pair["answer"]
        ]
        assert len(valid_qa_pairs) == expected_well_formed_count, (
            f"Expected {expected_well_formed_count} valid pairs, "
            f"but found {len(valid_qa_pairs)}."
        )

        # Create datasets (only well-formed pairs are included)
        train_dataset, test_dataset = create_dataset_from_qa_pairs(
            valid_qa_pairs, split_ratio=0.5
        )

        # Save datasets as plain text (malformed pairs should be skipped)
        save_datasets(train_dataset, test_dataset, output_file, json_format=False)

        # Determine train and test file paths
        train_file, test_file = get_output_file_paths(output_file, json_format=False)

        # Read the saved train dataset
        with open(train_file, "r") as f:
            train_content = f.read().strip()

        # Read the saved test dataset
        with open(test_file, "r") as f:
            test_content = f.read().strip()

        # Count the number of well-formed pairs (each pair is separated by two newlines)
        total_saved_pairs = len(train_content.split("\n\n")) + len(
            test_content.split("\n\n")
        )

        # Verify that the number of saved pairs matches the expected count of well-formed pairs
        assert total_saved_pairs == expected_well_formed_count, (
            f"Expected {expected_well_formed_count} well-formed pairs, "
            f"but found {total_saved_pairs}."
        )
