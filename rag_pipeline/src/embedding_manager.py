from sentence_transformers import SentenceTransformer

class EmbeddingManager:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def get_embeddings(self, chunks):
        # Check if there are any chunks
        if not chunks:
            print("No chunks found! Exiting.")
            return []
        try:
            embeddings = self.model.encode([chunk['page_content'] for chunk in chunks])
            print(f"Generated embeddings")
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return []
        return embeddings