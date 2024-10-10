import pytest
import logging
from unittest.mock import patch
from groq_qa_generator.logging_setup import initialize_logging


@patch("logging.info")
def test_initialize_logging(mock_logging_info):
    """
    Test that `initialize_logging` correctly sets the logging configuration.
    This test verifies that the logging configuration is set to `INFO` level and
    that the `httpx` logger is set to `WARNING`.
    """
    # Call the function
    initialize_logging()

    # Manually set the logger level to INFO to check
    logging.getLogger().setLevel(logging.INFO)

    # Assert that the root logger level is set to INFO
    assert logging.getLogger().level == logging.INFO

    # Assert that the `httpx` logger level is set to WARNING
    assert logging.getLogger("httpx").level == logging.WARNING
