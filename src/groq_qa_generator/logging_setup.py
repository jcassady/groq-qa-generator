import logging


def initialize_logging():
    """Configure the logging settings for the application.

    This function sets up the global logging configuration to display log messages
    at the `INFO` level or higher. The log format includes the timestamp, logger name,
    log level, and message. It also adjusts the log level for the 'httpx' library to
    display only warnings or higher to reduce unnecessary verbosity.

    Logging format:
    - Timestamp (in the format 'YYYY-MM-DD HH:MM:SS')
    - Logger name
    - Log level (e.g., INFO, WARNING)
    - Log message
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set 'httpx' logging level to WARNING to suppress Groq's INFO level logging.
    logging.getLogger("httpx").setLevel(logging.WARNING)
