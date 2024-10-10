import pytest
from groq_qa_generator.tokenizer import count_tokens, generate_text_chunks

# A longer excerpt from "Alice's Adventures in Wonderland"
ALICE_TEXT = (
    "Alice was beginning to get very tired of sitting by her sister on the bank, "
    "and of having nothing to do: once or twice she had peeped into the book her sister was reading, "
    "but it had no pictures or conversations in it, 'and what is the use of a book,' thought Alice, "
    "'without pictures or conversations?' So she was considering in her own mind (as well as she could, "
    "for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy chain "
    "would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink "
    "eyes ran close by her. There was nothing very remarkable in that, nor did Alice think it so very much out "
    "of the way to hear the Rabbit say to itself, 'Oh dear! Oh dear! I shall be late!' (when she thought it "
    "over afterwards, it occurred to her that she ought to have wondered at this, but at the time it all seemed "
    "quite natural); but when the Rabbit actually took a watch out of its waistcoat pocket, and looked at it, "
    "and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before "
    "seen a rabbit with either a waistcoat pocket, or a watch to take out of it. And burning with curiosity, "
    "she ran across the field after it, and fortunately was just in time to see it pop down a large rabbit hole, "
    "under the hedge."
)


def test_count_tokens():
    """
    Test the `count_tokens` function to ensure it correctly counts the number of tokens in the given text.

    The test covers two cases:
    1. A simple text ("Hello, world!") to verify that the function can handle short, straightforward inputs.
    2. A longer excerpt from "Alice's Adventures in Wonderland" to check if the function can accurately
       count tokens in larger, more complex inputs.

    We use `pytest.mark.parametrize` to run multiple test cases with expected token counts.

    Token counting is based on the function's internal tokenization mechanism, which may differ slightly
    from common word counting, particularly when handling punctuation, contractions, and whitespace.
    """

    # Test cases with expected token counts
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("Hello, world!", 4),  # A simple example to verify basic token counting
            (
                ALICE_TEXT,
                139,  # The token count expected for the longer Alice excerpt (adjust based on tokenizer)
            ),
        ],
    )
    def test_count_tokens(text, expected):
        """
        Assert that the `count_tokens` function returns the correct number of tokens
        for various types of input texts, ranging from simple to complex.
        
        Args:
            text (str): The input text for which token count is being checked.
            expected (int): The expected number of tokens for the given text.

        Returns:
            None
        """
        assert count_tokens(text) == expected
