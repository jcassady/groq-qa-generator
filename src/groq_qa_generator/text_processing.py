import re
import json
import logging
import os


def clean_text(text):
    """Cleans the input text by removing excessive whitespace.

    This function replaces all sequences of whitespace characters (including tabs,
    newlines, and multiple spaces) with a single space. It also trims any leading
    or trailing whitespace from the text.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned text with excessive whitespace removed and leading/trailing
             whitespace trimmed.
    """
    return re.sub(r"\s+", " ", text).strip()


def write_response_to_file(response, output_file, json_format=False):
    """
    Write the generated response to the specified output file.

    Depending on the `json_format` flag and the type of `response`, the response is
    either written as JSON or plain text.

    Args:
        response (str or list of dict): The response to be written to the file.
            - If `json_format` is True, `response` should be a list of dictionaries.
            - If `json_format` is False, `response` can be a string or a list of dictionaries.
        output_file (str): The base name for the output file (with extension).
        json_format (bool): Flag to indicate whether to write as JSON. Defaults to False.

    Side Effects:
        Writes the response to the specified output file.
    """

    def write_to_json(qa_pairs, json_file_path):
        """
        Write the response to a JSON file, handling any existing data.

        Args:
            qa_pairs (list of dict): The list of question-answer dictionaries to be written.
            json_file_path (str): The path to the JSON file.

        Side Effects:
            Updates the JSON file with new question-answer pairs.
        """
        # Load existing JSON data or start fresh if needed
        existing_data = load_existing_json_data(json_file_path)

        # Append new QA pairs to the existing data
        existing_data.extend(qa_pairs)

        # Write the updated data back to the JSON file
        try:
            with open(json_file_path, "w", encoding="utf-8") as json_file:
                json.dump(existing_data, json_file, ensure_ascii=False, indent=4)
            logging.info(
                f"QA pairs successfully written to JSON file: {json_file_path}"
            )
        except Exception as e:
            logging.error(f"Failed to write QA pairs to JSON file: {e}")

    def load_existing_json_data(json_file_path):
        """
        Load existing data from a JSON file, or return an empty list if there are issues.

        Args:
            json_file_path (str): The path to the JSON file.

        Returns:
            list: Existing data from the JSON file, or an empty list if the file is empty or invalid.
        """
        if os.path.exists(json_file_path):
            try:
                with open(json_file_path, "r", encoding="utf-8") as json_file:
                    data = json.load(json_file)
                    if isinstance(data, list):
                        logging.info(
                            f"Loaded {len(data)} existing QA pairs from {json_file_path}."
                        )
                        return data
                    else:
                        logging.warning(
                            f"Unexpected JSON structure in {json_file_path}. Starting fresh."
                        )
            except json.JSONDecodeError:
                logging.warning(
                    f"JSON decode error in {json_file_path}, starting fresh."
                )
            except Exception as e:
                logging.error(f"Error loading JSON data from {json_file_path}: {e}")
        else:
            logging.info(
                f"JSON file {json_file_path} does not exist. Starting with an empty QA list."
            )
        return []

    def write_to_text(qa_pairs, text_file_path):
        """
        Write the response to a text file, handling any existing data.

        Args:
            qa_pairs (list of dict): The list of question-answer dictionaries to be written.
            text_file_path (str): The path to the text file.

        Side Effects:
            Updates the text file with new question-answer pairs.
        """
        # Load existing QA pairs from the text file, if any
        existing_qa_pairs = load_existing_text_data(text_file_path)

        # Convert new QA pairs to string format
        new_qa_pairs = [f"{qa['question']}\n{qa['answer']}" for qa in qa_pairs]

        # Append new QA pairs to existing ones
        existing_qa_pairs.extend(new_qa_pairs)

        # Join all QA pairs with double newlines
        content = "\n\n".join(existing_qa_pairs)

        # Write back to text file
        try:
            with open(text_file_path, "w", encoding="utf-8") as f:
                f.write(content)
            logging.info(
                f"QA pairs successfully written to text file: {text_file_path}"
            )
        except Exception as e:
            logging.error(f"Failed to write QA pairs to text file: {e}")

    def write_to_text_from_string(response_str, text_file_path):
        """
        Write the response string to a text file.

        Args:
            response_str (str): The response string to be written.
            text_file_path (str): The path to the text file.

        Side Effects:
            Overwrites the content of the text file with the provided response string.
        """
        try:
            with open(text_file_path, "w", encoding="utf-8") as text_file:
                text_file.write(response_str)
            logging.info(
                f"Response successfully written to text file: {text_file_path}"
            )
        except Exception as e:
            logging.error(f"Failed to write response to text file: {e}")

    def load_existing_text_data(text_file_path):
        """
        Load existing QA pairs from a text file, or return an empty list if the file doesn't exist.

        Each QA pair is expected to be separated by double newlines, and within each pair,
        the question and answer are separated by a single newline.

        Args:
            text_file_path (str): The path to the text file.

        Returns:
            list of str: A list of "question\nanswer" strings.
        """
        if os.path.exists(text_file_path):
            try:
                with open(text_file_path, "r", encoding="utf-8") as text_file:
                    content = text_file.read().strip()
                    if content:
                        qa_pairs = content.split("\n\n")
                        logging.info(
                            f"Loaded {len(qa_pairs)} existing QA pairs from {text_file_path}."
                        )
                        return qa_pairs
                    else:
                        logging.info(
                            f"No existing QA pairs found in {text_file_path}. Starting fresh."
                        )
            except Exception as e:
                logging.error(f"Error reading {text_file_path}: {e}")
        else:
            logging.info(
                f"Text file {text_file_path} does not exist. Starting with an empty QA list."
            )
        return []

    def save_text_data(text_file_path, qa_pairs):
        """
        Save the provided QA pairs to a text file.

        Args:
            text_file_path (str): The path to the text file.
            qa_pairs (list): The list of "question\nanswer" strings to be written.

        Side Effects:
            Overwrites the content of the text file with the provided QA pairs.
        """
        try:
            with open(text_file_path, "w", encoding="utf-8") as text_file:
                for qa in qa_pairs:
                    text_file.write(qa + "\n\n")
            logging.info(
                f"QA pairs successfully written to text file: {text_file_path}"
            )
        except Exception as e:
            logging.error(f"Failed to write QA pairs to text file: {e}")

    def parse_response_into_qa_pairs(response):
        """
        Parse the response into question-answer pairs.

        Args:
            response (str): The response string to be parsed.

        Returns:
            list: A list of question-answer pair strings, each in the format "question\nanswer".
        """
        return response.strip().split("\n\n")

    # Log the response being processed
    logging.info("Writing response to file.")

    # Determine the format for writing the response
    if json_format:
        if isinstance(response, list) and all(isinstance(qa, dict) for qa in response):
            # Ensure the output file has a .json extension
            json_file_path = (
                output_file if output_file.endswith(".json") else f"{output_file}.json"
            )
            write_to_json(response, json_file_path)
        else:
            logging.error(
                "When json_format is True, response must be a list of dictionaries."
            )
    else:
        if isinstance(response, list) and all(isinstance(qa, dict) for qa in response):
            # Convert list of dicts to "question\nanswer" format and append
            write_to_text(response, output_file)
        elif isinstance(response, str):
            # Write the raw string to the text file
            write_to_text_from_string(response, output_file)
        else:
            logging.error(
                "When json_format is False, response must be a string or a list of dictionaries."
            )
