from langchain.chains import LLMChain
from recommender_pipeline.prompts import (
                                            #market_trends_prompt,
                                          skill_mapping_prompt,
                                          income_comparison_prompt, career_recommendation_prompt, upskilling_prompt)

# def create_market_trends_chain(llm):
#     return LLMChain(llm=llm, prompt=market_trends_prompt, output_key="market_explanation")

def create_skill_mapping_chain(llm):
    return LLMChain(llm=llm, prompt=skill_mapping_prompt, output_key="skill_mapping")

def create_income_comparison_chain(llm):
    return LLMChain(llm=llm, prompt=income_comparison_prompt, output_key="income_comparison")

def create_career_recommendation_chain(llm):
    return LLMChain(llm=llm, prompt=career_recommendation_prompt, output_key="career_recommendation")

def create_upskilling_chain(llm):
    return LLMChain(llm=llm, prompt=upskilling_prompt, output_key="upskilling_directions")
