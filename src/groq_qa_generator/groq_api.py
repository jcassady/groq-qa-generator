from groq import Groq
import logging
import os
from dotenv import load_dotenv
from .logging_setup import initialize_logging


def get_api_key():
    """Retrieve the Groq API key from environment variables.

    This function loads environment variables from a .env file (if present)
    and returns the value of the GROQ_API_KEY variable.

    Returns:
        str: The Groq API key.

    Raises:
        Exception: If the GROQ_API_KEY variable is not set in the environment.
    """
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise Exception("GROQ_API_KEY must be set in the environment variables.")

    return api_key


def get_groq_client(api_key):
    """Initialize and return a Groq API client.

    Args:
        api_key (str): The API key used to authenticate with the Groq service.

    Returns:
        Groq: An instance of the Groq client for interacting with the Groq API.
    """
    return Groq(api_key=api_key)


def get_groq_completion(
    client, system_prompt, chunk_text, model, temperature, max_tokens
):
    """Generate a completion from the Groq API using a system prompt and input text.

    This function sends a request to the Groq API to generate a completion based
    on the provided system prompt and chunked input text.

    Args:
        client (Groq): The Groq API client.
        system_prompt (str): The prompt that defines the system's behavior for the model.
        chunk_text (str): The input text chunk that is being processed by the model.
        model (str): The model identifier (e.g., "llama3-70b-8192") for the completion.
        temperature (float): The temperature setting to control randomness in the output.
        max_tokens (int): The maximum number of tokens the model can generate in the response.

    Returns:
        completion (object): The Groq API response object containing the completion results.
        None: If an error occurs during the API call, logs the error and returns None.
    """
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk_text},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        return completion
    except Exception as e:
        logging.error(f"Error with Groq API call: {e}")
        return None


def stream_completion(completion):
    """Stream the Groq API completion and return the accumulated response.

    This function streams the generated completion from the Groq API.

    Args:
        completion (object): The streamed response from the Groq API.

    Returns:
        str: The accumulated response from the streamed completion.
    """
    response = ""
    for data_chunk in completion:
        content = data_chunk.choices[0].delta.content or ""
        response += content

    return response
