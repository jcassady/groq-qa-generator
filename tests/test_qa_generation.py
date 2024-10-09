import pytest
import logging
from unittest.mock import patch, mock_open
from groq_qa_generator.qa_generation import (
    load_sample_question,
    load_system_prompt,
    create_groq_prompt,
)


def test_load_sample_question():
    """Test loading a sample question from a file.

    This test checks that the load_sample_question function correctly reads
    the content of a file and returns it as a stripped string. It uses
    mock_open to simulate file reading without creating an actual file.

    It verifies that the output matches the expected question content.
    """
    sample_content = (
        "Would you like an adventure now, or shall we have our tea first?\n"
    )
    with patch("builtins.open", mock_open(read_data=sample_content)):
        result = load_sample_question("sample_question.txt")
    assert result == "Would you like an adventure now, or shall we have our tea first?"


def test_load_system_prompt():
    """Test loading and preparing the system prompt.

    This test verifies that the load_system_prompt function reads the system
    prompt from a file and replaces the "<n>" placeholder with the correct
    number of questions based on the provided chunk size and tokens per question.

    It ensures that the returned prompt contains the expected format and
    structure after processing.
    """
    system_prompt_content = "Generate <n> questions based on the text."
    chunk_size = 512
    tokens_per_question = 60
    expected_prompt = "Generate 8 questions based on the text."

    with patch("builtins.open", mock_open(read_data=system_prompt_content)):
        result = load_system_prompt(
            "system_prompt.txt", chunk_size, tokens_per_question
        )
    assert result == expected_prompt


def test_create_groq_prompt():
    """Test the creation of the Groq prompt.

    This test checks that the create_groq_prompt function combines the
    system prompt and sample question correctly into a full prompt.

    It ensures that the output prompt contains both the system prompt and
    the sample question in the expected format.
    """
    system_prompt = "Generate <n> questions based on the text."
    sample_question = "Why is a raven like a writing desk?"
    expected_full_prompt = f"{system_prompt}\n\n{sample_question}"

    result = create_groq_prompt(system_prompt, sample_question)
    assert result == expected_full_prompt
