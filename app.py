import gradio as gr
from main import initialize_model, generate_career_recommendations, save_recommendations_to_file
from utils.config_loader import load_config, load_env_variables


def gradio_recommendation_interface(user_skills, current_income):
    """Interface function for Gradio."""
    # Load environment variables and initialize model
    env_vars = load_env_variables()
    api_token = env_vars["huggingface_api_token"]
    llm = initialize_model("config/config.yaml", api_token)

    # Prepare user input
    user_input = {
        "user_skills": user_skills.split(", "),
        "current_income": float(current_income)
    }

    # Generate recommendations
    results = generate_career_recommendations(
        llm,
        "data/output/rag_output.json",
        "data/pay_info.json",
        user_input
    )

    # Optionally save results
    save_recommendations_to_file(results, 'data/output/career_recommendations.json')

    return results


# Define the Gradio app
inputs = [
    gr.Textbox(label="Enter your skills (comma-separated)", placeholder="e.g., Python, SQL, Machine Learning"),
    gr.Textbox(label="Enter your current income in $/h", placeholder="e.g., 30"),
]

outputs = gr.JSON(label="Career Recommendations")

app = gr.Interface(
    fn=gradio_recommendation_interface,
    inputs=inputs,
    outputs=outputs,
    title="Career Recommendation Chatbot",
    description="Get career advice based on your skills and income.",
)

if __name__ == "__main__":
    app.launch()
