from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from recommender_pipeline.prompts import (market_trends_prompt, skill_mapping_prompt,
                                          income_comparison_prompt, career_recommendation_prompt, upskilling_prompt)


def create_market_trends_chain(llm):
    return LLMChain(llm=llm, prompt=market_trends_prompt, output_key="market_explanation")


def create_skill_mapping_chain(llm):
    return LLMChain(llm=llm, prompt=skill_mapping_prompt, output_key="skill_mapping")


def create_income_comparison_chain(llm):
    return LLMChain(llm=llm, prompt=income_comparison_prompt, output_key="income_comparison")


def create_career_recommendation_chain(llm):
    return LLMChain(llm=llm, prompt=career_recommendation_prompt, output_key="career_recommendation")


def create_upskilling_chain(llm):
    return LLMChain(llm=llm, prompt=upskilling_prompt, output_key="upskilling_directions")


def create_pipeline(llm):
    market_trends_chain = create_market_trends_chain(llm)
    skill_mapping_chain = create_skill_mapping_chain(llm)
    income_comparison_chain = create_income_comparison_chain(llm)
    career_recommendation_chain = create_career_recommendation_chain(llm)
    upskilling_chain = create_upskilling_chain(llm)

    return SequentialChain(
        chains=[
            market_trends_chain, skill_mapping_chain,
            income_comparison_chain, career_recommendation_chain, upskilling_chain
        ],
        input_variables=["market_trends", "user_skills", "current_income", "market_income"],
        output_variables=["market_explanation", "skill_mapping", "income_comparison", "career_recommendation",
                          "upskilling_directions"]
    )
