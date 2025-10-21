from crewai import Agent

def create_feature_engineer_agent(llm):
    """Agent responsible for feature engineering and selection"""
    return Agent(
        role='Feature Engineering Specialist',
        goal='Create and select the most relevant features for machine learning models',
        backstory='''You are an expert in feature engineering with deep knowledge of 
        feature creation, selection, and transformation techniques. You understand 
        which features are important for different types of problems and can create 
        polynomial features, interaction terms, and apply appropriate transformations. 
        You also know how to handle categorical variables and scale numerical features.''',
        llm=llm,
        verbose=True,
        allow_delegation=False
    )