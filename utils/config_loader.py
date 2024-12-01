import os
import yaml
from dotenv import load_dotenv

def load_config(config_path):
    """
    Load the configuration from a YAML file.
    Args:
        config_path (str): Path to the configuration file.
    Returns:
        dict: Configuration data.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_env_variables():
    """
    Load environment variables from the .env file or system environment.
    Returns:
        dict: Environment variables required for the pipeline.
    """
    load_dotenv()  # Load from .env if available
    env_vars = {
        "huggingface_api_token": os.getenv("HUGGINGFACE_API_TOKEN"),
    }

    # Validate that all required variables are set
    if not env_vars["huggingface_api_token"]:
        raise EnvironmentError("Missing Hugging Face API token. Please set it in .env or system variables.")

    return env_vars
