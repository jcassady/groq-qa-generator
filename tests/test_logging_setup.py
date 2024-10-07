import logging
import pytest
from groq_qa_generator.logging_setup import initialize_logging


def test_setup_logging_configures_logging_level():
    """
    Test that setup_logging configures the global logging level to INFO.

    This test verifies that after calling the setup_logging function, the global
    logging level is set to INFO, which allows for INFO level and higher messages
    to be displayed.
    """
    initialize_logging()
    assert logging.getLogger().level == logging.INFO


def test_setup_logging_configures_logging_format():
    """
    Test that setup_logging configures the logging format correctly.

    This test checks that the logging format is set to include the timestamp,
    logger name, log level, and message, formatted according to the specified
    settings in the setup_logging function.
    """
    initialize_logging()

    # Ensure that the logger has handlers
    assert len(logging.getLogger().handlers) > 0, "Logger has no handlers configured."

    # Check the format of the first handler's formatter
    log_format = logging.getLogger().handlers[0].formatter._fmt
    expected_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Check that the actual format matches the expected format
    assert (
        log_format == expected_format
    ), f"Expected format: {expected_format}, but got: {log_format}"


def test_setup_logging_sets_httpx_logging_level():
    """
    Test that setup_logging sets the httpx logging level to WARNING.

    This test verifies that after calling the setup_logging function, the logging
    level for the 'httpx' logger is set to WARNING, ensuring that only warning
    and higher messages from the 'httpx' library are displayed.
    """
    initialize_logging()
    httpx_logger = logging.getLogger("httpx")
    assert httpx_logger.level == logging.WARNING
