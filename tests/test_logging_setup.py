import logging
import pytest
from groq_qa_generator.logging_setup import initialize_logging


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
