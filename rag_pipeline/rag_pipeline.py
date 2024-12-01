from rag_pipeline.document_processor import load_documents, preprocess_documents
from rag_pipeline.embedding_manager import EmbeddingManager
from rag_pipeline.chromadb_manager import ChromaDBManager
from rag_pipeline.generate_response import ResponseGenerator
import json
import os


class RAGPipeline:
    def __init__(self, embedding_model_name, rag_llm_model_name, api_token, max_token_limit, chunk_overlap,
                 collection_name="tech_jobs"):
        self.api_token = api_token
        self.collection_name = collection_name
        self.max_token_limit = max_token_limit
        self.chunk_overlap = chunk_overlap

        # Initialize individual managers
        self.embedding_manager = EmbeddingManager(embedding_model_name)
        self.chromadb_manager = ChromaDBManager(api_token, collection_name)
        self.response_generator = ResponseGenerator(rag_llm_model_name, api_token)

    def run(self, documents_path, query_prompt):
        # Load and process documents
        documents = load_documents(documents_path)
        chunks = preprocess_documents(documents, max_token_limit=self.max_token_limit, chunk_overlap=self.chunk_overlap)

        # Generate embeddings
        embeddings = self.embedding_manager.get_embeddings(chunks)

        # Build ChromaDB index
        collection = self.chromadb_manager.build_chromadb_index(chunks, embeddings)

        # Retrieve relevant chunks for the query
        retrieved_chunks = self.chromadb_manager.retrieve_relevant_chunks(query_prompt, collection,
                                                                          self.embedding_manager.model)

        # Generate response using the RAG model
        response = self.response_generator.generate_response(query_prompt, retrieved_chunks, self.max_token_limit)
        print(response)

        # Save the response to a file
        output_file_path = os.path.join("data/output/", 'rag_output.json')
        rag_output = {"market_trends": response}
        with open(output_file_path, 'w') as json_file:
            json.dump(rag_output, json_file)
        print("LLM output saved to rag_output.json")
