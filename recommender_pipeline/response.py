from utils import extract_answer

def format_response(results):
    """Format the results from each chain into a structured JSON-friendly format."""
    formatted_response = {}

    # Market explanation
    market_explanation = results.get("market_explanation", "No data available")
    formatted_response['Market Trends'] = {
        "Overview": extract_answer(market_explanation)
    }

    # Skill mapping
    skill_mapping = results.get("skill_mapping", "No data available")
    formatted_response['Skill Mapping'] = {
        "Required Skills": extract_answer(skill_mapping)
    }

    # Income comparison
    income_comparison = results.get("income_comparison", "No data available")
    formatted_response['Income Comparison'] = {
        "Comparison": extract_answer(income_comparison)
    }

    # Career recommendations
    career_recommendation = results.get("career_recommendation", "No data available")
    formatted_response['Career Recommendations'] = {
        "Suggested Career Paths": extract_answer(career_recommendation)
    }

    # Upskilling directions
    upskilling_directions = results.get("upskilling_directions", "No data available")
    formatted_response['Upskilling Directions'] = {
        "Recommendations": extract_answer(upskilling_directions)
    }

    return formatted_response