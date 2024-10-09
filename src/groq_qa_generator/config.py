import os
import argparse
import json
import shutil
import logging


def parse_arguments():
    """
    Parse command-line arguments for generating QA pairs from text input.

    This function defines and parses the available command-line arguments, including:
    - `--json`: A flag to specify whether the output should be saved in JSON format.
    - `--model`: The model to be used for generating QA pairs (default: "llama3-70b-8192").
    - `--temperature`: A floating-point value controlling the randomness of model output (default: 0.1).

    The function also validates the `temperature` argument to ensure it falls within the valid range of 0.0 to 1.0.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.

    Raises:
        SystemExit: If there is an error in argument parsing or temperature validation.

    Example usage:
        python your_script.py --json --model "some_model" --temperature 0.5
    """

    try:

        def validate_temperature(temperature):
            """
            Validate that the temperature is within the range [0.0, 1.0].

            Args:
                temperature (float): The temperature value to validate.

            Raises:
                ValueError: If the temperature is not between 0.0 and 1.0.
            """
            if not (0.0 <= temperature <= 1.0):
                raise ValueError("Temperature must be between 0.0 and 1.0.")

        # Create the argument parser
        parser = argparse.ArgumentParser(
            description="Generate QA pairs from text input."
        )

        # Add argument for JSON output
        parser.add_argument(
            "--json",
            action="store_true",
            default=False,
            help="Save QA pairs as JSON instead of plain text.",
        )

        # Add argument for the model to use
        parser.add_argument(
            "--model",
            type=str,
            default="llama3-70b-8192",  # Default model
            help="Specify the model to be used for generating QA pairs.",
        )

        # Add argument for temperature setting
        parser.add_argument(
            "--temperature",
            type=float,
            default=0.1,  # Default temperature
            help="Set the temperature for the model's output generation. Must be between 0.0 and 1.0.",
        )

        parser.add_argument(
            "--questions",
            type=int,
            default=None,
            help="Number of QA pairs per text chunk to generate.",
        )

        # Parse the arguments
        args = parser.parse_args()

        # Validate the temperature argument
        validate_temperature(args.temperature)

        return args

    except argparse.ArgumentError as e:
        """
        Handle argument parsing errors.

        Args:
            e (ArgumentError): The exception raised during argument parsing.

        Side Effects:
            Prints the error message and exits the program.
        """
        print(f"Argument parsing error: {e}")
        raise SystemExit("Error: Invalid arguments provided.")


def load_config(args, config_file="config.json"):
    """
    Load the configuration file, resolve file paths, and override settings with CLI arguments if provided.

    This function loads the configuration settings from a JSON file, resolves file paths based on the
    user configuration directory, and applies any command-line argument overrides provided by the user
    via `args`.

    Args:
        args (argparse.Namespace): Command-line arguments parsed by the `parse_arguments` function.
                                   These can be used to override configuration file values.
        config_file (str): The name of the configuration file. Defaults to "config.json".

    Returns:
        dict: The loaded configuration dictionary with resolved paths and overrides applied.

    Raises:
        FileNotFoundError: If the configuration file cannot be found.
        ValueError: If there is an error parsing the JSON configuration file.
        SystemExit: If there is an issue loading or processing the configuration file.
    """

    def get_user_config_dir():
        """Get the path to the user's configuration directory."""
        return os.path.expanduser("~/.groq_qa/")

    def load_json_config(config_file):
        """
        Load the configuration from a JSON file.

        Args:
            config_file (str): The name of the configuration file.

        Returns:
            dict: The loaded configuration data.

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If the config file contains invalid JSON.
        """
        config_file_path = os.path.join(get_user_config_dir(), config_file)
        if not os.path.exists(config_file_path):
            raise FileNotFoundError(f"Config file not found: {config_file_path}")
        try:
            with open(config_file_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON configuration: {e}")

    def resolve_config_paths(config):
        """
        Resolve the paths for system prompt, sample question, input data, and output file.

        Args:
            config (dict): The loaded configuration dictionary.

        Returns:
            dict: The updated configuration with resolved paths.

        Raises:
            KeyError: If any expected keys are missing in the config dictionary.
        """
        user_config_dir = get_user_config_dir()
        config["system_prompt"] = os.path.join(
            user_config_dir, "data", "prompts", config["system_prompt"]
        )
        config["sample_question"] = os.path.join(
            user_config_dir, "data", "prompts", config["sample_question"]
        )
        config["input_data"] = os.path.join(
            user_config_dir, "data", config["input_data"]
        )
        config["output_file"] = os.path.join(user_config_dir, config["output_file"])
        return config

    def override_config_with_cli_args(args, config):
        """
        Override configuration values with command-line arguments if provided.

        Args:
            config (dict): The configuration dictionary to update.

        Side Effects:
            Updates the config dictionary with CLI arguments if they are provided.

        Raises:
            KeyError: If any expected keys for CLI arguments are missing in the config dictionary.
        """
        # Set model and temperature if provided, otherwise keep existing config
        config["model"] = args.model or config.get("model", "default_model")
        config["temperature"] = args.temperature if args.temperature is not None else config.get("temperature", 0.1)

        # Set questions with a default value of None if not provided
        config["questions"] = args.questions or None

        # Set JSON output handling, modifying the output file if JSON is selected
        if args.json:
            base_output_file = os.path.splitext(config["output_file"])[0]
            config["output_file"] = f"{base_output_file}.json"
        config["json"] = args.json or False


    try:
        config = load_json_config(config_file)
        config = resolve_config_paths(config)
        override_config_with_cli_args(args, config)
        log_config(config, os.path.join(get_user_config_dir(), config_file))
        return config
    except FileNotFoundError as e:
        logging.error(f"Configuration file error: {e}")
        raise SystemExit(f"Error: {e}")
    except ValueError as e:
        logging.error(f"Configuration parsing error: {e}")
        raise SystemExit(f"Error: {e}")


def log_config(config, config_file_path):
    """
    Log the configuration to the console in a readable format.

    Args:
        config (dict): The configuration dictionary to log.
        config_file_path (str): The file path of the configuration file.

    Raises:
        ValueError: If the provided config is not a dictionary.
    """

    def validate_config_is_dict(config):
        """Ensure the configuration is a dictionary."""
        if not isinstance(config, dict):
            raise ValueError("Invalid config format: expected a dictionary.")

    def prepare_log_formatting(config):
        """Prepare the header and formatting for logging the configuration."""
        max_key_length = max(len(key) for key in config.keys()) + 2
        header = f"{'Configuration Key'.ljust(max_key_length)} | Value"
        separator = "-" * len(header)
        return header, separator, max_key_length

    try:
        validate_config_is_dict(config)
        header, separator, max_key_length = prepare_log_formatting(config)
        logging.info(f"\nConfig file path: {config_file_path}")
        logging.info(header)
        logging.info(separator)
        for key, value in config.items():
            logging.info(f"{key.ljust(max_key_length)} | {value}")
        logging.info("")
    except ValueError as e:
        logging.error(f"Error in configuration logging: {e}")
        raise SystemExit(f"Error: {e}")


def initialize_user_config():
    """
    Set up the user-specific configuration directory and copy necessary files.

    This function creates the user configuration directory (if not already present)
    and copies necessary files (such as config.json and data directories) into the
    user configuration directory.

    Raises:
        FileNotFoundError: If the source files to be copied do not exist.
        PermissionError: If the program lacks permission to create directories or copy files.
        SystemExit: If any fatal errors occur during the setup process.
    """

    def create_directory(directory):
        """
        Create the specified directory if it doesn't exist.

        Args:
            directory (str): The path of the directory to create.

        Raises:
            PermissionError: If the program lacks permission to create the directory.
        """
        try:
            os.makedirs(directory, exist_ok=True)
            logging.debug(f"User configuration directory created: {directory}")
        except PermissionError as e:
            logging.error(f"Permission error: {e}")
            raise SystemExit(
                f"Error: Insufficient permissions to create or modify files in {directory}."
            )

    def copy_directory(src, dest, label):
        """
        Copy the directory from the source to the destination.

        Args:
            src (str): The source directory path.
            dest (str): The destination directory path.
            label (str): A label for logging purposes.

        Raises:
            FileNotFoundError: If the source directory does not exist.
            shutil.Error: If there is an error during the directory copying process.
        """
        try:
            if os.path.exists(dest):
                logging.debug(
                    f"{label} directory already exists at: {dest}. Skipping copy."
                )
            else:
                shutil.copytree(src, dest)
                logging.debug(f"Copied {label} directory from {src} to {dest}")
        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            raise SystemExit(f"Error: {e}")
        except shutil.Error as e:
            logging.error(f"File copy error: {e}")
            raise SystemExit(f"Error: Failed to copy files or directories: {e}")

    def get_package_data_path():
        """Return the path to the package's data directory."""
        return os.path.join(get_package_dir(), "data")

    def get_package_dir():
        """Return the directory path of the current package."""
        return os.path.dirname(__file__)

    def copy_config_file_if_missing(user_config_dir):
        """
        Copy the configuration file to the user directory if it doesn't already exist.

        Args:
            user_config_dir (str): The user's configuration directory.

        Raises:
            FileNotFoundError: If the source config.json file is not found.
            PermissionError: If the program lacks permission to copy the file.
        """
        try:
            config_src_path = os.path.join(get_package_dir(), "config.json")
            config_dest_path = os.path.join(user_config_dir, "config.json")
            if not os.path.exists(config_dest_path):
                shutil.copy(config_src_path, config_dest_path)
                logging.debug(
                    f"Copied config.json from {config_src_path} to {config_dest_path}"
                )
            else:
                logging.debug(
                    f"Config file already exists at: {config_dest_path}. Skipping copy."
                )
        except FileNotFoundError as e:
            logging.error(f"Config file not found: {e}")
            raise SystemExit(f"Error: {e}")
        except PermissionError as e:
            logging.error(f"Permission error: {e}")
            raise SystemExit(
                f"Error: Insufficient permissions to copy config file: {e}"
            )

    user_config_dir = os.path.expanduser("~/.groq_qa/")
    create_directory(user_config_dir)
    copy_directory(
        get_package_data_path(), os.path.join(user_config_dir, "data"), "Data"
    )
    copy_config_file_if_missing(user_config_dir)
