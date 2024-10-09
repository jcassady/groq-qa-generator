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

    Depending on the `json_format` flag, the response is either written as JSON or plain text.
    
    Args:
        response (str): The response string to be written to the file.
        output_file (str): The base name for the output file (without extension).
        json_format (bool): Flag to indicate whether to write as JSON. Defaults to False.

    Side Effects:
        Writes the response to the specified output file.
    """

    def write_to_json(response, json_file_path):
        """
        Write the response to a JSON file, handling any existing data.

        Args:
            response (str): The response string to be written.
            json_file_path (str): The path to the JSON file.

        Side Effects:
            Updates the JSON file with new question-answer pairs.
        """
        # Load existing JSON data or start fresh if needed
        response_data = load_existing_json_data(json_file_path)

        # Parse the response into question-answer pairs
        qa_pairs = parse_response_into_qa_pairs(response)

        # Append parsed QA pairs to the JSON data
        for qa in qa_pairs:
            if "\n" in qa:
                question, answer = qa.split("\n", 1)
                response_data.append({"question": question.strip(), "answer": answer.strip()})
            else:
                logging.warning(f"Malformed QA pair found: {qa}")

        # Write the updated data back to the JSON file
        save_json_data(json_file_path, response_data)

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
                    return json.load(json_file)
            except json.JSONDecodeError:
                logging.warning(f"JSON decode error in {json_file_path}, starting fresh.")
        return []

    def save_json_data(json_file_path, data):
        """
        Save the provided data to a JSON file.

        Args:
            json_file_path (str): The path to the JSON file.
            data (list): The list of question-answer pairs to be written to the file.

        Side Effects:
            Overwrites the content of the JSON file with the provided data.
        """
        with open(json_file_path, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)

    def write_to_text(response, text_file_path):
        """
        Write the response to a text file, handling any existing data.

        Args:
            response (str): The response string to be written.
            text_file_path (str): The path to the text file.

        Side Effects:
            Updates the text file with new question-answer pairs.
        """
        # Load existing QA pairs from the text file, if any
        existing_qa_pairs = load_existing_text_data(text_file_path)

        # Parse the response into question-answer pairs
        qa_pairs = parse_response_into_qa_pairs(response)

        # Append parsed QA pairs to the existing pairs
        for qa in qa_pairs:
            if "\n" in qa:
                question, answer = qa.split("\n", 1)
                existing_qa_pairs.append(f"{question.strip()}\n{answer.strip()}")
            else:
                logging.warning(f"Malformed QA pair found: {qa}")

        # Write all QA pairs (existing + new) to the text file
        save_text_data(text_file_path, existing_qa_pairs)

    def load_existing_text_data(text_file_path):
        """
        Load existing QA pairs from a text file, or return an empty list if the file doesn't exist.

        Args:
            text_file_path (str): The path to the text file.

        Returns:
            list: Existing QA pairs, or an empty list if the file is empty or invalid.
        """
        if os.path.exists(text_file_path):
            with open(text_file_path, "r", encoding="utf-8") as text_file:
                return text_file.read().strip().split("\n\n")
        return []

    def save_text_data(text_file_path, qa_pairs):
        """
        Save the provided QA pairs to a text file.

        Args:
            text_file_path (str): The path to the text file.
            qa_pairs (list): The list of question-answer pairs to be written.

        Side Effects:
            Overwrites the content of the text file with the provided QA pairs.
        """
        with open(text_file_path, "w", encoding="utf-8") as text_file:
            for qa in qa_pairs:
                text_file.write(qa + "\n\n")

    def parse_response_into_qa_pairs(response):
        """
        Parse the response into question-answer pairs.

        Args:
            response (str): The response string to be parsed.

        Returns:
            list: A list of question-answer pairs.
        """
        return response.strip().split("\n\n")
    
    # Log the response being processed
    logging.info(response)

    # Determine the format for writing the response
    if json_format:
        json_file_path = output_file.replace(".txt", ".json")
        write_to_json(response, json_file_path)
    else:
        text_file_path = output_file
        write_to_text(response, text_file_path)
