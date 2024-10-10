import logging
import textwrap


def initialize_logging():
    """Configure the logging settings for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)


def qa_table(title, qa_pairs, question_width=40, answer_width=40):
    """
    Logs a formatted ASCII table of QA pairs, with text wrapping for long content.
    Adds a numbered column and a total count of QA pairs at the bottom of the table.
    Each row is visually separated with dashed lines, including the numbers column.
    The title is included at the top of the table.

    Args:
        title (str): Title for the table (e.g., "Training QA Pairs").
        qa_pairs (list of dict): List of QA pairs to be displayed in the table.
        question_width (int): Maximum width for the question column.
        answer_width (int): Maximum width for the answer column.
    """
    # Calculate the total table width including the new column for numbering and borders
    num_column_width = 5  # width of the numbering column, including separators
    question_column_width = question_width + 2  # question width + padding for borders
    answer_column_width = answer_width + 2  # answer width + padding for borders
    total_width = num_column_width + question_column_width + answer_column_width

    # Adjust the title box to fit the title text and left-align it
    title_box_width = len(title) + 2  # +2 for the padding
    title_box = f"+{'-' * title_box_width}+"
    title_text = f"| {title} |"

    # Log the title box
    logging.info(title_box)
    logging.info(title_text)
    logging.info(title_box)

    # Log the column headers
    logging.info(
        f"+{'-' * num_column_width}+{'-' * question_column_width}+{'-' * answer_column_width}+"
    )
    logging.info(
        f"| {'#':<3} | {'Question':<{question_width}} | {'Answer':<{answer_width}} |"
    )
    logging.info(
        f"+{'-' * num_column_width}+{'-' * question_column_width}+{'-' * answer_column_width}+"
    )

    # Log each QA pair
    for idx, pair in enumerate(qa_pairs, start=1):
        question = textwrap.wrap(pair["question"], width=question_width)
        answer = textwrap.wrap(pair["answer"], width=answer_width)

        # Determine the maximum number of lines between the question and answer
        max_lines = max(len(question), len(answer))

        # Pad the shorter list with empty strings so both lists have the same number of lines
        question.extend([""] * (max_lines - len(question)))
        answer.extend([""] * (max_lines - len(answer)))

        # Log each line of the question and answer
        for i, (q, a) in enumerate(zip(question, answer)):
            if i == 0:
                logging.info(
                    f"| {idx:<3} | {q:<{question_width}} | {a:<{answer_width}} |"
                )
            else:
                logging.info(
                    f"| {' ':<3} | {q:<{question_width}} | {a:<{answer_width}} |"
                )

        # Add dashed separator below each row except the last one
        if idx < len(qa_pairs):
            logging.info(
                f"| {'-' * 3:<3} | {'-' * question_width:<{question_width}} | {'-' * answer_width:<{answer_width}} |"
            )

    # Final separator and total QA Pairs count
    logging.info(
        f"+{'-' * num_column_width}+{'-' * question_column_width}+{'-' * answer_column_width}+"
    )

    total_qa_text = "Total QA Pairs:"
    padding_required = (
        total_width - len(total_qa_text) - len(str(len(qa_pairs))) - 3
    )  # Adjust for borders

    # Log the "Total QA Pairs" line with proper alignment
    logging.info(f"| {total_qa_text} {' ' * padding_required}{len(qa_pairs)}   |")
    logging.info(
        f"+{'-' * num_column_width}+{'-' * question_column_width}+{'-' * answer_column_width}+\n"
    )
