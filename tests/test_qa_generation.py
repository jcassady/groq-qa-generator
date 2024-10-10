import logging
import os
import pytest
from unittest.mock import patch, MagicMock
from groq_qa_generator.qa_generation import generate_qa_pairs


@patch("groq_qa_generator.qa_generation.get_groq_completion")
@patch("groq_qa_generator.qa_generation.stream_completion")
@patch("groq_qa_generator.qa_generation.get_groq_client")
@patch("groq_qa_generator.qa_generation.get_api_key")
def test_generate_qa_pairs_with_alice_samples(
    mock_get_api_key,
    mock_get_groq_client,
    mock_stream_completion,
    mock_get_groq_completion,
):
    """
    Test the `generate_qa_pairs` function using sample text from Alice in Wonderland.

    This test simulates the process of generating question-answer pairs by calling the Groq API.
    The function is mocked to ensure that it behaves as expected and returns the correct output.
    """

    # Set the API key programmatically in the environment for this test
    os.environ["GROQ_API_KEY"] = "gsk_test_fake_key"

    # Mocked API key and client initialization
    mock_get_api_key.return_value = os.getenv("GROQ_API_KEY")
    mock_get_groq_client.return_value = MagicMock()

    # Mocked responses from the Groq API (questions and answers based on Alice in Wonderland)
    mock_get_groq_completion.return_value = "mock-completion-object"
    mock_stream_completion.return_value = (
        "What did Alice find on the three-legged glass table that gave her hope of escaping the hall?\n"
        "A tiny golden key that might unlock one of the doors in the hall.\n\n"
        "What was Alice worried about when she thought about her cat Dinah while falling down the hole?\n"
        "Alice was worried that Dinah would miss her and hoped someone would remember to give Dinah her saucer of milk at tea-time.\n\n"
        "What was Alice's plan when she found the small cake with the words 'EAT ME' on it?\n"
        "Alice planned to eat the cake, hoping it would make her grow larger to reach the key or smaller to creep under the door, so she could get into the garden.\n\n"
        "What did Alice notice as she fell down the well, and what did she do with a jar she took down from one of the shelves?\n"
        "She saw cupboards, book-shelves, maps, and pictures on the sides of the well, and she took down an empty jar labelled 'ORANGE MARMALADE' and put it back into a cupboard as she continued to fall."
    )

    # Mock configuration for the Groq API and QA generation process
    groq_config = {
        "system_prompt": "system_prompt.txt",
        "sample_question": "sample_question.txt",
        "chunk_size": 512,
        "tokens_per_question": 60,
        "model": "groq-model",
        "temperature": 0.5,
        "max_tokens": 100,
        "questions": None,
        "output_file": "output.txt",
        "json": False,
    }

    # Sample text chunk that will be processed to generate QA pairs
    text_chunks = ["This is a sample text chunk from Alice in Wonderland."]

    # Mock the `load_system_prompt` and `load_sample_question` functions for the test
    with patch(
        "groq_qa_generator.qa_generation.load_system_prompt",
        return_value="Sample system prompt",
    ):
        with patch(
            "groq_qa_generator.qa_generation.load_sample_question",
            return_value="Sample question",
        ):
            # Call the `generate_qa_pairs` function to process the mock input and generate QA pairs
            qa_pairs = generate_qa_pairs(text_chunks, groq_config)

    # Define the expected QA pairs based on the mocked Alice in Wonderland content
    expected_qa_pairs = [
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

    # Assert that the generated QA pairs match the expected number and content
    assert len(qa_pairs) == 4
    assert qa_pairs == expected_qa_pairs

    # Ensure that the mocked functions were called with the correct arguments
    mock_get_api_key.assert_called_once()
    mock_get_groq_client.assert_called_once_with("gsk_test_fake_key")
    mock_get_groq_completion.assert_called_once()
    mock_stream_completion.assert_called_once_with("mock-completion-object")
