from utils import load_json
from recommender_pipeline.pipeline import generate_recommendation
from recommender_pipeline.user_input import get_user_input
from utils.config_loader import load_config, load_env_variables
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
    # max_length = config["max_length"]
    max_new_tokens = config["max_new_tokens"]
    api_token = env_vars["huggingface_api_token"]

    # Run RAG pipeline
    rag_main()

    # Load data
    market_trends = load_json("data/output/rag_output.json")
    pay_info = load_json("data/pay_info.json")

    # Initialize LLM
    model = AutoModelForCausalLM.from_pretrained(llm_model_name, use_auth_token=api_token)
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name, use_auth_token=api_token)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    hf_pipeline = pipeline(
        'text-generation', model=model, tokenizer=tokenizer,
        device=0 if device == "cuda" else -1
        # , max_length=max_length
        , max_new_tokens=max_new_tokens
        ,temperature=0.3
        ,top_k=30
        ,top_p=0.8)

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # Get user input
    user_input = get_user_input()
    user_skills = user_input["user_skills"]
    current_income = user_input["current_income"]
    market_income = pay_info

    results = generate_recommendation(llm, market_trends, user_skills, current_income, market_income)

    print("Recommendation Results:")
    print(json.dumps(results, indent=4))

    with open('data\output\career_recommendations.json', 'w') as json_file:
        json.dump(results, json_file, indent=4)


if __name__ == "__main__":
    main()
