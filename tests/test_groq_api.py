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
    Test get_groq_client function to ensure it returns a valid client instance.

    This test checks if the Groq client is properly initialized using the given API key.
    """
    api_key = "gsk_test_1234567890abcdef"
    client = get_groq_client(api_key)
    assert client is not None


@patch("groq_qa_generator.groq_api.Groq")
def test_get_groq_completion(mock_groq):
    """
    Test get_groq_completion function with a valid client and inputs.

    This test checks if the completion function correctly calls the Groq API client
    and returns the expected completion result using the provided system prompt and chunk text.
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
    Test get_groq_completion function when an error occurs during the API call.

    This test simulates an exception being raised by the API client, ensuring
    that the function handles the error gracefully by logging it and returning None.
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
    Test stream_completion function to ensure the streaming response is properly accumulated.

    This test checks if the function correctly accumulates the content from the streamed
    Groq API response and returns the full generated text.
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
