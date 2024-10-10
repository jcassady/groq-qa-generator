import pytest
from unittest.mock import patch
from groq_qa_generator.config import parse_arguments


def test_parse_arguments_default(monkeypatch):
    """
    Test the default behavior of the `parse_arguments` function.

    This test ensures that when no command-line arguments are provided,
    the function assigns default values to all arguments:
    - `--json`: False (output not in JSON format)
    - `--model`: "llama3-70b-8192" (default model)
    - `--temperature`: 0.1 (default temperature setting)
    - `--questions`: None (no specified number of questions)
    - `--split`: 0.8 (80% training split by default)
    - `--upload`: None (no repository upload specified)
    """
    monkeypatch.setattr("sys.argv", ["groq-qa"])  # Simulate running with no arguments
    args = parse_arguments()

    # Assert default values for all arguments
    assert args.json is False
    assert args.model == "llama3-70b-8192"
    assert args.temperature == 0.1
    assert args.questions is None
    assert args.split == 0.8
    assert args.upload is None


def test_parse_arguments_json_flag(monkeypatch):
    """
    Test the `parse_arguments` function with the `--json` flag.

    This test verifies that when the `--json` flag is passed, it sets
    the `json` argument to True, indicating that the output should be saved
    in JSON format.
    """
    monkeypatch.setattr("sys.argv", ["groq-qa", "--json"])  # Simulate passing '--json'
    args = parse_arguments()
    assert args.json is True


def test_parse_arguments_with_questions(monkeypatch):
    """
    Test the `--questions` argument in `parse_arguments`.

    This test ensures that the `--questions` argument, which specifies the number
    of questions to generate, is correctly parsed and stored in the `questions` attribute
    of the parsed arguments object.
    """
    monkeypatch.setattr("sys.argv", ["groq-qa", "--questions", "1"])
    args = parse_arguments()

    # Assert that the questions argument is correctly parsed as an integer
    assert args.questions == 1


def test_parse_arguments_with_split(monkeypatch):
    """
    Test the `--split` argument in `parse_arguments`.

    This test checks that a valid `--split` argument, representing the fraction
    of the dataset to use for training (e.g., 0.7 for 70% train, 30% test), is
    parsed and stored correctly in the parsed arguments.
    """
    monkeypatch.setattr("sys.argv", ["groq-qa", "--split", "0.7"])
    args = parse_arguments()
    assert args.split == 0.7


def test_parse_arguments_invalid_split(monkeypatch):
    """
    Test the `--split` argument with an invalid value.

    This test ensures that providing an invalid value for the `--split` argument
    (outside the range 0.0 to 1.0) raises a `ValueError`, as expected.
    """
    monkeypatch.setattr("sys.argv", ["groq-qa", "--split", "1.5"])

    # Assert that a ValueError is raised with the appropriate message
    with pytest.raises(ValueError, match="--split must be a float between 0.0 and 1.0"):
        parse_arguments()


def test_parse_arguments_with_upload(monkeypatch):
    """
    Test the `--upload` argument in `parse_arguments`.

    This test verifies that the `--upload` argument, which specifies the Hugging Face
    repository to upload the dataset to, is correctly parsed and stored.
    """
    monkeypatch.setattr("sys.argv", ["groq-qa", "--upload", "test-user/test-dataset"])
    args = parse_arguments()

    # Assert that the upload argument is correctly parsed as a string
    assert args.upload == "test-user/test-dataset"


def test_parse_arguments_invalid_upload(monkeypatch):
    """
    Test the `--upload` argument with an invalid value.

    This test checks that providing an invalid value for the `--upload` argument
    (e.g., an incorrect format or a placeholder like 'username/dataset') raises
    a `ValueError` with an appropriate message.
    """
    monkeypatch.setattr("sys.argv", ["groq-qa", "--upload", "invalid_repo_path"])

    # Assert that a ValueError is raised due to invalid Hugging Face repo path
    with pytest.raises(ValueError, match="Invalid Hugging Face repo path"):
        parse_arguments()


def test_parse_arguments_temperature_valid(monkeypatch):
    """
    Test the `--temperature` argument with a valid value.

    This test ensures that when a valid temperature value (within the range 0.0 to 1.0)
    is passed via the `--temperature` argument, it is correctly parsed and stored
    in the parsed arguments.
    """
    monkeypatch.setattr("sys.argv", ["groq-qa", "--temperature", "0.7"])
    args = parse_arguments()

    # Assert that the temperature is correctly parsed
    assert args.temperature == 0.7


def test_parse_arguments_temperature_invalid(monkeypatch):
    """
    Test the `--temperature` argument with an invalid value.

    This test checks that providing an invalid value for the `--temperature` argument
    (outside the range 0.0 to 1.0) raises a `ValueError`, as expected.
    """
    monkeypatch.setattr("sys.argv", ["groq-qa", "--temperature", "1.5"])

    # Assert that a ValueError is raised with the appropriate message
    with pytest.raises(ValueError, match="Temperature must be between 0.0 and 1.0"):
        parse_arguments()