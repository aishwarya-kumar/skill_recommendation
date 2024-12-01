def extract_answer(text, delimiter="Answer:"):
    """
    Extract the answer portion from a given text based on the specified delimiter.
    Parameters:
    - text (str): The input text containing a question and answer.
    - delimiter (str): The keyword used to separate the question from the answer. Default is "Answer:".
    Returns:
    - str: The portion of the text after the delimiter, or the entire text if the delimiter is not found.
    """
    if delimiter in text:
        return text.split(delimiter, 1)[1].strip()
    return text.strip()
