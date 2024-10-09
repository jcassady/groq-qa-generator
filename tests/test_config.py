# import argparse
import json
import pytest
from unittest.mock import mock_open, patch
from groq_qa_generator.config import parse_arguments


def test_parse_arguments_default(monkeypatch):
    """
    Test parse_arguments when no arguments are provided.

    This ensures that the '--json' flag is False by default when not passed in.
    """
    monkeypatch.setattr(
        "sys.argv", ["groq-qa"]
    )  # Simulate running the script with no arguments
    args = parse_arguments()
    assert args.json is False


def test_parse_arguments_json_flag(monkeypatch):
    """
    Test parse_arguments when the '--json' flag is provided.

    This ensures that the '--json' flag is set to True when passed in.
    """
    monkeypatch.setattr(
        "sys.argv", ["groq-qa", "--json"]
    )  # Simulate running with '--json'
    args = parse_arguments()
    assert args.json is True

def test_parse_arguments_with_questions(monkeypatch):
    """
    Test that the --questions argument is correctly parsed when using the 'groq-qa' command.

    This test ensures that when the --questions argument is provided via the 'groq-qa' command
    line, it is correctly parsed by the parse_arguments function and stored in the args object.

    The test uses the `monkeypatch` fixture to mock the command-line arguments as if
    the script is executed with 'groq-qa --questions 1'.

    Steps:
    1. Mock the command-line input to simulate running 'groq-qa --questions 1'.
    2. Call the parse_arguments() function to simulate argument parsing.
    3. Assert that args.questions is equal to 1.

    Input:
        Command-line args: ["groq-qa", "--questions", "1"]

    Expected output:
        args.questions == 1

    Edge cases considered:
    - The --questions argument is correctly parsed as an integer.
    """
    # Mock the sys.argv to simulate the command-line input for 'groq-qa'
    monkeypatch.setattr("sys.argv", ["groq-qa", "--questions", "1"])
    
    # Parse the arguments
    args = parse_arguments()
    
    # Check that the questions argument is correctly parsed
    assert args.questions == 1




