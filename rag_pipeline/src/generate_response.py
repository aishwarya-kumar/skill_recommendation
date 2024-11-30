from transformers import AutoTokenizer, AutoModelForCausalLM

class ResponseGenerator:
    def __init__(self, rag_llm_model_name, api_token):
        self.tokenizer = AutoTokenizer.from_pretrained(rag_llm_model_name, use_auth_token=api_token)
        self.model = AutoModelForCausalLM.from_pretrained(rag_llm_model_name, use_auth_token=api_token)

    def generate_response(self, query, retrieved_chunks, max_new_tokens=512):
        prompt = f"""
        Query: {query}

        Relevant Information:
        {retrieved_chunks}

        Answer the query based on the relevant information provided above:
        Answer:
        """
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=4096)
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            temperature=0.7,
            top_k=50,
            top_p=0.9
        )
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        if "Answer:" in response:
            return response.split("Answer:")[-1].strip()
        return response.strip()
