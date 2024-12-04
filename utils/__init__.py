import json


def load_json(json_path):
    with open(json_path, 'r') as file:
        return json.load(file)

# def extract_answer(text, delimiter="Answer:"):
#     if delimiter in text:
#         return text.split(delimiter, 1)[1].strip()
#     return text.strip()


def extract_answer(text, delimiter="Answer:"):
    if delimiter in text:
        return text.rsplit(delimiter, 1)[1].strip()
    return text.strip()


def remove_repetitions(output):
    lines = output.split("\n")
    unique_lines = list(dict.fromkeys(lines))  # Retain order and remove duplicates
    return "\n".join(unique_lines)