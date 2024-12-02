from langchain.chains import SequentialChain
from recommender_pipeline.chains import (
    # create_market_trends_chain,
    create_skill_mapping_chain,
    create_income_comparison_chain, create_career_recommendation_chain,
    create_upskilling_chain)
from utils import extract_answer, load_json


def create_overall_chain(llm):
    # market_trends_chain = create_market_trends_chain(llm)
    skill_mapping_chain = create_skill_mapping_chain(llm)
    income_comparison_chain = create_income_comparison_chain(llm)
    career_recommendation_chain = create_career_recommendation_chain(llm)
    upskilling_chain = create_upskilling_chain(llm)

    overall_chain = SequentialChain(
        chains=[skill_mapping_chain, income_comparison_chain, career_recommendation_chain,
                upskilling_chain],
        input_variables=["market_trends","user_skills", "current_income", "market_income"],
        output_variables=["skill_mapping", "income_comparison", "career_recommendation",
                          "upskilling_directions"]
    )
    return overall_chain


def generate_recommendation(llm, market_trends,  user_skills, current_income, market_income):
    overall_chain = create_overall_chain(llm)

    response = overall_chain.call({
        "user_skills": user_skills,
        "current_income": current_income,
        "market_income": market_income
    })

    skill_mapping_answer = extract_answer(response["skill_mapping"], delimiter="Answer:")
    income_comparison_answer = extract_answer(response["income_comparison"], delimiter="Answer:")
    career_recommendation_answer = extract_answer(response["career_recommendation"], delimiter="Answer:")
    upskilling_answer = extract_answer(response["upskilling_directions"], delimiter="Answer:")

    results = {
        "skill_mapping": skill_mapping_answer,
        "income_comparison": income_comparison_answer,
        "career_recommendation": career_recommendation_answer,
        "upskilling_directions": upskilling_answer
    }

    # response = overall_chain({"market_trends": market_trends, "user_skills": user_skills,
    #                           "current_income": current_income, "market_income": market_income})
    #
    # # market_explanation = extract_answer(response["market_explanation"], delimiter="Answer:")
    # skill_mapping = extract_answer(response["skill_mapping"], delimiter="Answer:")
    # income_comparison = extract_answer(response["income_comparison"], delimiter="Answer:")
    # career_recommendation = extract_answer(response["career_recommendation"], delimiter="Answer:")
    # upskilling_directions = extract_answer(response["upskilling_directions"], delimiter="Answer:")
    #
    # results = {"skill_mapping": skill_mapping,
    #            "income_comparison": income_comparison,
    #            "career_recommendation": career_recommendation,
    #            "upskilling_directions": upskilling_directions}
    # #"market_explanation": market_explanation,

    return results
