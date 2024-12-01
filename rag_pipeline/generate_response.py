from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import extract_answer

class ResponseGenerator:
    def __init__(self, rag_llm_model_name, api_token):
        self.tokenizer = AutoTokenizer.from_pretrained(rag_llm_model_name, use_auth_token=api_token)
        self.model = AutoModelForCausalLM.from_pretrained(rag_llm_model_name, use_auth_token=api_token)

    def generate_response(self, query, retrieved_chunks, max_new_tokens):
        prompt = f"""
        You are an AI assistant. Based on the information provided, answer the query concisely and directly.

        Query: {query}

        Relevant Information:
        {retrieved_chunks}

        Answer:
        """
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=4096)
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            temperature=0.3,
            top_k=20,
            top_p=0.7
        )
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        formatted_response = extract_answer(response, prompt)

        return formatted_response
