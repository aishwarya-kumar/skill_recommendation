import json

def load_json(json_path):
    with open(json_path, 'r') as file:
        return json.load(file)

# def extract_answer(text, delimiter="Answer:"):
#     if delimiter in text:
#         return text.split(delimiter, 1)[1].strip()
#     return text.strip()


# def extract_answer(response, prompt):
#     # Remove the prompt from the response
#     answer = response.replace(prompt, "").strip()
#     return answer


def extract_answer(response, prompt=None, delimiter="Answer:"):
    if prompt:
        # If prompt is provided, remove it from the response
        answer = response.replace(prompt, "").strip()
    elif delimiter in response.strip():
        # If no prompt, extract based on the delimiter
        answer = response.split(delimiter, 1)[1].strip()
    else:
        # If neither is available, return the response as-is
        answer = response.strip()
    return answer
