from langchain.prompts import PromptTemplate


# market_trends_prompt = PromptTemplate(
#     input_variables=["market_trends"],
#     template="""You are a recommendation system for gig workers and freelancers related to their career and skills. Users need advice
# their current skills, and potential career switch as freelancers in the tech industry. You need to analyze their current
# profile and compare it to the market trends. Market trends includes high paying and in-demand tech careers for
# freelancers and the skills required for these careers. The following are the latest market trends: {market_trends}
# Explain to the user the top 3 in-demand job roles and the top 5 skills needed for each role in a structured and clear
# concise manner.
# Answer:""")

skill_mapping_prompt = PromptTemplate(
    input_variables=["user_skills", "market_trends"],
    template="""You are a recommendation system for gig workers and freelancers related to their career and skills. Users need advice 
their current skills, and potential career switch as freelancers in the tech industry. You need to analyze their current 
profile and compare it to the market trends. Market trends includes high paying and in-demand tech careers for 
freelancers and the skills required for these careers.
    The user has the following skills: {user_skills}.
Based on the market trends provided: {market_trends},
1. Identify the top in-demand skills the user already has.
2. Map the user's skills to transferable skills for the top job roles.
Answer:""")

income_comparison_prompt = PromptTemplate(
    input_variables=["current_income", "market_income"],
    template="""The user's current income is {current_income} USD per hour.
The market average incomes for the top job roles are as follows: {market_income}
Identify the roles with higher income potential than the user's current role.
Answer:"""
)

career_recommendation_prompt = PromptTemplate(
    input_variables=["skill_mapping", "income_comparison"],
    template="""Based on the skill mapping results:
{skill_mapping}
And the income comparison results:
{income_comparison}
Recommend the top career role the user should switch to. Identify the skills needed for this role.
Answer:"""
)

upskilling_prompt = PromptTemplate(
    input_variables=["career_recommendation"],
    template="""The user should switch to the following career role: {career_recommendation}

Provide a detailed guide on how the user can obtain the required skills for this career switch.
Answer:"""
)
