def get_user_input():
    user_skills = input("Enter your current skills (comma-separated): ")
    current_income = input("Enter your current hourly income (in USD): ")
    return {"user_skills": user_skills, "current_income": current_income}
