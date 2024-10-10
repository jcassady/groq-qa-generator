import os
import pytest
from unittest.mock import MagicMock, patch
from groq_qa_generator.groq_api import (
    get_api_key,
    get_groq_client,
    get_groq_completion,
    stream_completion,
)


def test_get_groq_client():
    """
    Test the `get_groq_client` function to ensure it returns a valid Groq client instance.

    This test verifies that the Groq client is properly initialized when provided with a valid API key.
    It ensures that the client object returned is not None and can be used for making API calls.
    
    Asserts:
        - The client object returned is not None.
    """
    api_key = "gsk_test_1234567890abcdef"
    client = get_groq_client(api_key)
    assert client is not None


@patch("groq_qa_generator.groq_api.Groq")
def test_get_groq_completion(mock_groq):
    """
    Test the `get_groq_completion` function to ensure it generates valid completions using the Groq API.

    This test mocks the Groq API client and verifies that the `get_groq_completion` function correctly 
    calls the API with the provided system prompt, chunk of text, model, temperature, and token limit.
    It checks that the function returns the expected completion result.

    Args:
        mock_groq (MagicMock): Mock of the Groq API client.

    Asserts:
        - The completion result matches the expected mock value ("test_completion").
        - The API client is called once with the correct parameters.
    """
    mock_client = MagicMock()
    mock_groq.return_value = mock_client
    mock_client.chat.completions.create.return_value = "test_completion"

    system_prompt = "Generate <n> question and answer pairs based on the provided text."
    chunk_text = "Why, sometimes I've believed as many as six impossible things before breakfast."
    model = "llama3-70b-8192"
    temperature = 0.7
    max_tokens = 100

    completion = get_groq_completion(
        mock_client, system_prompt, chunk_text, model, temperature, max_tokens
    )

    assert completion == "test_completion"
    mock_client.chat.completions.create.assert_called_once_with(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chunk_text},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )


@patch("groq_qa_generator.groq_api.Groq")
def test_get_groq_completion_error(mock_groq):
    """
    Test the `get_groq_completion` function to handle API errors gracefully.

    This test simulates an error raised during the API call and ensures that the function catches
    the exception, logs the error, and returns `None` instead of raising the exception.

    Args:
        mock_groq (MagicMock): Mock of the Groq API client that simulates an error.

    Asserts:
        - The function returns `None` when an exception is raised.
    """
    mock_client = MagicMock()
    mock_groq.return_value = mock_client
    mock_client.chat.completions.create.side_effect = Exception("API error")

    text = "Why, sometimes I've believed as many as six impossible things before breakfast."
    completion = get_groq_completion(
        mock_client, "System Prompt", text, "llama3-70b-8192", 0.7, 100
    )

    assert completion is None


def test_stream_completion():
    """
    Test the `stream_completion` function to ensure it accumulates the streamed response correctly.

    This test verifies that the `stream_completion` function correctly processes and combines 
    the incremental content chunks streamed from the Groq API. It checks that the full generated 
    text is accumulated and returned as a single string.

    Asserts:
        - The function correctly concatenates and returns the complete response from the stream.
    """
    mock_completion = [
        MagicMock(
            choices=[
                MagicMock(
                    delta=MagicMock(content="Why, sometimes I've believed as many as ")
                )
            ]
        ),
        MagicMock(
            choices=[
                MagicMock(
                    delta=MagicMock(content="six impossible things before breakfast.")
                )
            ]
        ),
    ]

    result = stream_completion(mock_completion)
    assert (
        result
        == "Why, sometimes I've believed as many as six impossible things before breakfast."
    )
