from utils import load_json
from recommender_pipeline.pipeline import generate_recommendation
from recommender_pipeline.user_input import get_user_input
from utils.config_loader import load_config, load_env_variables
from rag_pipeline.main import main as rag_main
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
import torch
import json


def initialize_model(config_path, api_token):
    config = load_config(config_path)
    llm_model_name = config["llm_model_name"]
    max_new_tokens = config["max_new_tokens"]

    model = AutoModelForCausalLM.from_pretrained(llm_model_name, use_auth_token=api_token)
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name, use_auth_token=api_token)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    hf_pipeline = pipeline(
        'text-generation', model=model, tokenizer=tokenizer,
        device=0 if device == "cuda" else -1,
        max_new_tokens=max_new_tokens,
        temperature=0.3,
        top_k=30,
        top_p=0.8
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    return llm


def generate_career_recommendations(llm, market_trends_path, pay_info_path, user_input):
    """Generate recommendations based on market trends and user input."""
    market_trends = load_json(market_trends_path)
    pay_info = load_json(pay_info_path)

    user_skills = user_input["user_skills"]
    current_income = user_input["current_income"]
    market_income = pay_info

    results = generate_recommendation(llm, market_trends, user_skills, current_income, market_income)
    return results


def save_recommendations_to_file(results, output_path):
    """Save the generated recommendations to a JSON file."""
    with open(output_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)


def main():
    # Load configuration and environment variables
    config = load_config("config/config.yaml")
    env_vars = load_env_variables()
    api_token = env_vars["huggingface_api_token"]

    # Run RAG pipeline
    rag_main()

    # Initialize LLM
    llm = initialize_model("config/config.yaml", api_token)

    # Get user input
    user_input = get_user_input()

    # Generate recommendations
    results = generate_career_recommendations(
        llm,
        "data/output/rag_output.json",
        "data/pay_info.json",
        user_input
    )

    print("Recommendation Results:")
    print(json.dumps(results, indent=4))

    # Save recommendations
    save_recommendations_to_file(results, 'data/output/career_recommendations.json')


if __name__ == "__main__":
    main()
