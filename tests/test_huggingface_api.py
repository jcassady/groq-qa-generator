import pytest
from unittest.mock import patch, MagicMock
from datasets import Dataset

from groq_qa_generator.huggingface_api import upload


@pytest.fixture
def mock_train_dataset():
    """
    Fixture to create a mock Hugging Face Dataset representing the training set.

    This mock object simulates the behavior of a Hugging Face `Dataset` and is
    used in tests to validate interactions with the training dataset without needing
    an actual dataset.

    Returns:
        MagicMock: A mock object simulating the Hugging Face `Dataset`.
    """
    return MagicMock(spec=Dataset)


@pytest.fixture
def mock_test_dataset():
    """
    Fixture to create a mock Hugging Face Dataset representing the test set.

    This mock object simulates the behavior of a Hugging Face `Dataset` and is
    used in tests to validate interactions with the test dataset without needing
    an actual dataset.

    Returns:
        MagicMock: A mock object simulating the Hugging Face `Dataset`.
    """
    return MagicMock(spec=Dataset)


@pytest.fixture
def mock_repo():
    """
    Fixture to provide a sample Hugging Face repository name for use in tests.

    This fixture is used to simulate a valid repository path during tests where
    dataset uploads are performed.

    Returns:
        str: A sample Hugging Face repository name.
    """
    return "test-user/test-repo"


@patch("groq_qa_generator.huggingface_api.DatasetDict.push_to_hub")
def test_upload_to_hub_with_repo(
    mock_push_to_hub, mock_train_dataset, mock_test_dataset, mock_repo
):
    """
    Test the `upload` function when a valid Hugging Face repository is provided.

    This test ensures that when a valid repository name is passed to the `upload` function,
    the `push_to_hub` method is called with the correct repository and datasets.

    Args:
        mock_push_to_hub (MagicMock): Mocked method for pushing datasets to Hugging Face hub.
        mock_train_dataset (MagicMock): Mocked Hugging Face train dataset fixture.
        mock_test_dataset (MagicMock): Mocked Hugging Face test dataset fixture.
        mock_repo (str): Mock repository name fixture.

    Asserts:
        - The `push_to_hub` method is called exactly once with the repository name and private flag.
    """
    # Act: Call the upload function with the mocked inputs
    upload(mock_train_dataset, mock_test_dataset, mock_repo)

    # Assert: Ensure push_to_hub is called with the correct arguments
    mock_push_to_hub.assert_called_once_with(mock_repo, private=True)


@patch("groq_qa_generator.huggingface_api.logging.warning")
def test_upload_without_repo_logs_warning(
    mock_logging_warning, mock_train_dataset, mock_test_dataset
):
    """
    Test the `upload` function when no repository is provided, ensuring it logs a warning.

    This test verifies that if no repository is passed to the `upload` function, it does not
    attempt to push the datasets to Hugging Face. Instead, it logs a warning message and exits.

    Args:
        mock_logging_warning (MagicMock): Mocked logging method for capturing warning messages.
        mock_train_dataset (MagicMock): Mocked Hugging Face train dataset fixture.
        mock_test_dataset (MagicMock): Mocked Hugging Face test dataset fixture.

    Asserts:
        - A warning is logged indicating that the repository was not provided.
    """
    # Act: Call the upload function without providing a repository
    upload(mock_train_dataset, mock_test_dataset, repo=None)

    # Assert: Ensure a warning was logged
    mock_logging_warning.assert_called_once_with(
        "Hugging Face Hub repository not provided. Skipping QA dataset upload."
    )


@patch("groq_qa_generator.huggingface_api.DatasetDict.push_to_hub")
@patch("groq_qa_generator.huggingface_api.logging.info")
def test_upload_logging_info_on_success(
    mock_logging_info,
    mock_push_to_hub,
    mock_train_dataset,
    mock_test_dataset,
    mock_repo,
):
    """
    Test the `upload` function to verify logging of success messages after a successful upload.

    This test ensures that when the `upload` function successfully uploads the datasets to
    Hugging Face using a valid repository, it logs appropriate info messages indicating the
    start and success of the upload process.

    Args:
        mock_logging_info (MagicMock): Mocked logging method for capturing info messages.
        mock_push_to_hub (MagicMock): Mocked method for pushing datasets to Hugging Face hub.
        mock_train_dataset (MagicMock): Mocked Hugging Face train dataset fixture.
        mock_test_dataset (MagicMock): Mocked Hugging Face test dataset fixture.
        mock_repo (str): Mock repository name fixture.

    Asserts:
        - An info log is generated indicating that the upload process has started.
        - An info log is generated with the URL to the uploaded dataset.
    """
    # Act: Call the upload function with the mocked inputs
    upload(mock_train_dataset, mock_test_dataset, mock_repo)

    # Assert: Ensure that an info log was generated for the successful upload
    mock_logging_info.assert_any_call("Uploading QA dataset to Hugging Face Hub.")
    mock_logging_info.assert_any_call(
        f"Dataset uploaded to Hugging Face hub at https://huggingface.co/datasets/{mock_repo}"
    )


@patch("groq_qa_generator.huggingface_api.DatasetDict")
def test_upload_to_huggingface(mock_dataset_dict):
    """
    Test the `upload` function to ensure it correctly uploads datasets to the Hugging Face Hub.

    This test mocks the Hugging Face `DatasetDict` and verifies that:
    - The dataset is correctly uploaded when a valid repository is provided.
    - The function logs a warning and skips the upload if no repository is provided.

    Args:
        mock_dataset_dict (MagicMock): Mock for the Hugging Face `DatasetDict`.
    """
    # Mock train and test datasets
    mock_train_dataset = MagicMock()
    mock_test_dataset = MagicMock()

    # Mock the DatasetDict object and its method `push_to_hub`
    mock_dataset_instance = MagicMock()
    mock_dataset_dict.return_value = mock_dataset_instance

    # Case 1: Valid repository provided
    repo = "mock-repo"
    upload(mock_train_dataset, mock_test_dataset, repo)

    # Ensure DatasetDict was created with train and test datasets
    mock_dataset_dict.assert_called_once_with(
        {"train": mock_train_dataset, "eval": mock_test_dataset}
    )

    # Ensure `push_to_hub` was called with the correct repository
    mock_dataset_instance.push_to_hub.assert_called_once_with(repo, private=True)

    # Reset the mock for the next case
    mock_dataset_dict.reset_mock()
    mock_dataset_instance.push_to_hub.reset_mock()

    # Case 2: No repository provided
    upload(mock_train_dataset, mock_test_dataset, None)

    # Ensure that `push_to_hub` is not called when repo is None
    mock_dataset_instance.push_to_hub.assert_not_called()
