from unittest.mock import patch
from groq_qa_generator.groq_qa import generate


@patch("groq_qa_generator.groq_qa.upload")
@patch("groq_qa_generator.groq_qa.create_dataset_from_qa_pairs")
@patch("groq_qa_generator.groq_qa.generate_qa_pairs")
@patch("groq_qa_generator.groq_qa.generate_text_chunks")
@patch("groq_qa_generator.groq_qa.initialize_logging")
def test_generate_function(
    mock_initialize_logging,
    mock_generate_text_chunks,
    mock_generate_qa_pairs,
    mock_create_dataset_from_qa_pairs,
    mock_upload,
):
    """
    Test the `generate` function to ensure it processes the configuration correctly,
    generates QA pairs, and uploads them to Hugging Face.

    This test mocks the internal calls to:
    - `initialize_logging`: To verify logging initialization is called.
    - `generate_text_chunks`: To simulate text chunking.
    - `generate_qa_pairs`: To simulate QA pair generation.
    - `create_dataset_from_qa_pairs`: To simulate dataset creation and splitting.
    - `upload`: To mock the uploading process to Hugging Face Hub.

    Assertions:
    - Logging is initialized correctly.
    - Text chunks are generated from the provided input file.
    - QA pairs are generated based on the text chunks.
    - Datasets are split and uploaded with the correct repo.

    """
    # Mock input and output for the functions
    mock_generate_text_chunks.return_value = ["chunk1", "chunk2"]
    mock_generate_qa_pairs.return_value = [
        {"question": "What is...", "answer": "This is..."}
    ]
    mock_create_dataset_from_qa_pairs.return_value = ("train_dataset", "test_dataset")

    # Sample config to pass to the generate function
    config = {
        "system_prompt": "mock_system_prompt.txt",
        "sample_question": "mock_sample_question.txt",
        "input_data": "mock_input_data.txt",
        "output_file": "mock_output.txt",
        "model": "llama3-70b-8192",
        "chunk_size": 512,
        "tokens_per_question": 60,
        "temperature": 0.7,
        "max_tokens": 1024,
        "split_ratio": 0.8,
        "huggingface_repo": "username/dataset",
    }

    # Call the generate function with the mocked config
    generate(config)

    # Verify that logging was initialized
    mock_initialize_logging.assert_called_once()

    # Verify that text chunks were generated from the input data
    mock_generate_text_chunks.assert_called_once_with(
        "mock_input_data.txt", chunk_size=512
    )

    # Capture the actual config passed to `generate_qa_pairs`
    actual_call_args = mock_generate_qa_pairs.call_args[0]
    text_chunks_passed = actual_call_args[0]
    config_passed = actual_call_args[1]

    # Verify that QA pairs were generated using the correct text chunks
    assert text_chunks_passed == ["chunk1", "chunk2"]

    # Verify that the config values passed to `generate_qa_pairs` match except for the `huggingface_repo`
    expected_config = {
        "system_prompt": "mock_system_prompt.txt",
        "sample_question": "mock_sample_question.txt",
        "input_data": "mock_input_data.txt",
        "output_file": "mock_output.txt",
        "model": "llama3-70b-8192",
        "chunk_size": 512,
        "tokens_per_question": 60,
        "temperature": 0.7,
        "max_tokens": 1024,
        "split_ratio": 0.8,
    }

    # Check if the passed config matches the expected config excluding huggingface_repo
    for key, value in expected_config.items():
        assert config_passed[key] == value

    # Special check for huggingface_repo
    assert "huggingface_repo" in config_passed
    assert (
        config_passed["huggingface_repo"] == config["huggingface_repo"]
        or "username/dataset"
    )

    # Verify that datasets were created from the QA pairs with the correct split ratio
    mock_create_dataset_from_qa_pairs.assert_called_once_with(
        [{"question": "What is...", "answer": "This is..."}], 0.8
    )

    # Verify that the upload function was called with the correct parameters
    mock_upload.assert_called_once_with(
        "train_dataset", "test_dataset", config["huggingface_repo"]
    )
