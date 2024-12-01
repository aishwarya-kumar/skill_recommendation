import chromadb

class ChromaDBManager:
    def __init__(self, api_token, collection_name="tech_jobs"):
        self.api_token = api_token
        self.client = chromadb.Client()
        self.collection_name = collection_name

    def build_chromadb_index(self, documents, embeddings):
        collections = self.client.list_collections()
        collection_names = [collection.name for collection in collections]

        if self.collection_name in collection_names:
            self.client.delete_collection(name=self.collection_name)
            print(f"Deleted existing collection '{self.collection_name}'.")

        collection = self.client.create_collection(name=self.collection_name)
        print(f"Created a new collection '{self.collection_name}'.")

        documents_text = [doc['page_content'] for doc in documents]
        if not documents_text:
            raise ValueError("No valid text content found in documents.")

        collection.add(
            documents=documents_text,
            embeddings=embeddings,
            ids=[str(i) for i in range(len(documents))]
        )
        print(f"Added {len(documents)} documents to ChromaDB collection.")
        return collection

    def retrieve_relevant_chunks(self, query, collection, embedding_model):
        query_embedding = embedding_model.encode([query]).tolist()
        query_result = collection.query(query_embeddings=query_embedding, n_results=3)
        return query_result['documents']
