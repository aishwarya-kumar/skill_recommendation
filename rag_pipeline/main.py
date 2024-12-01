from datetime import datetime
from rag_pipeline.rag_pipeline import RAGPipeline
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config_loader import load_config, load_env_variables


def main():
    # Load configurations
    config = load_config("config/config.yaml")

    # Load environment variables
    env_vars = load_env_variables()

    # Initialize RAG pipeline
    rag_pipeline = RAGPipeline(
        embedding_model_name=config["embedding_model_name"],
        rag_llm_model_name=config["rag_llm_model_name"],
        api_token=env_vars["huggingface_api_token"],
        max_token_limit=config["max_token_limit"],
        chunk_overlap=config["chunk_overlap"],
        collection_name=config["chromadb_collection_name"]
    )
    # Get the current year and month
    current_year = datetime.now().year
    current_month = datetime.now().strftime("%B")

    # Define path to documents and query prompt
    documents_path = "data/rag_documents/"
    query_prompt = config["query_prompt"]
    query_prompt = query_prompt.format(current_month=current_month, current_year=current_year)

    # Run the pipeline
    rag_pipeline.run(documents_path, query_prompt)

if __name__ == "__main__":
    main()
