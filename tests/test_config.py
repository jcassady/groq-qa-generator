# import argparse
import json
import pytest
from unittest.mock import mock_open, patch
from groq_qa_generator.config import parse_arguments, load_config


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


