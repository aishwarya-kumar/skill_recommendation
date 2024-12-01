from recommender_pipeline.data_loader import load_market_trends, load_pay_info
from recommender_pipeline.pipeline import create_pipeline
from recommender_pipeline.user_input import get_user_input
from utils.config_loader import load_config, load_env_variables
from recommender_pipeline.response import format_response
from rag_pipeline.main import main as rag_main
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
import torch
import json


def main():
    # Load configuration from the config file
    config = load_config("config/config.yaml")

    env_vars = load_env_variables()

    llm_model_name = config["llm_model_name"]
    max_length = config["max_length"]
    max_new_tokens = config["max_new_tokens"]
    api_token = env_vars["huggingface_api_token"]

    # Run RAG pipeline
    rag_main()

    # Load data
    market_trends = load_market_trends("data/output/rag_output.json")
    pay_info = load_pay_info("data/pay_info.csv")

    # Initialize LLM
    model = AutoModelForCausalLM.from_pretrained(llm_model_name, use_auth_token=api_token)
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name, use_auth_token=api_token)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    hf_pipeline = pipeline(
        'text-generation', model=model, tokenizer=tokenizer,
        device=0 if device == "cuda" else -1, max_length=max_length, max_new_tokens=max_new_tokens)

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # Create pipeline
    recommender_pipeline = create_pipeline(llm)

    # Get user input
    user_input = get_user_input()
    user_input["market_trends"] = market_trends
    user_input["market_income"] = pay_info.to_dict()

    # Run pipeline
    results = recommender_pipeline.invoke(user_input)
    # print("Results:", results)

    formatted_response = format_response(results)

    with open('data\output\career_recommendations.json', 'w') as json_file:
        json.dump(formatted_response, json_file, indent=4)

    print("\nCareer Recommendations and Insights:")
    for key, value in formatted_response.items():
        print(f"\n{key}:")
        for subkey, subvalue in value.items():
            print(f"  {subkey}: {subvalue}")


if __name__ == "__main__":
    main()
