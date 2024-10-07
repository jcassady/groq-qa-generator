import tiktoken


def count_tokens(text):
    """Counts the number of tokens in a given text using the specified tokenization model.

    This function uses the "cl100k_base" encoding model from `tiktoken` to encode the input
    text and counts the total number of tokens generated.

    Args:
        text (str): The input text for which tokens need to be counted.

    Returns:
        int: The total number of tokens in the input text.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    return sum(1 for _ in encoding.encode(text))


def generate_text_chunks(file_path, chunk_size):
    """Reads text from a file and splits it into chunks based on a token limit.

    This function reads the input text file line by line, and accumulates text into
    chunks such that the total number of tokens in each chunk does not exceed `chunk_size`.
    When the token limit is reached, the chunk is added to the list of chunks.

    Args:
        file_path (str): The path to the text file that needs to be chunked.
        chunk_size (int): The maximum number of tokens allowed in each chunk.

    Returns:
        list of str: A list containing text chunks where each chunk respects the token limit.
    """
    chunks = []
    chunk = ""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            # If adding the current line exceeds the chunk size, save the current chunk
            if count_tokens(chunk + text) > chunk_size:
                chunks.append(chunk.strip())
                chunk = text
            else:
                # Append the line to the current chunk
                chunk += " " + text
    # Append any remaining chunk
    if chunk:
        chunks.append(chunk.strip())
    return chunks
