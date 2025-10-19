from crewai import Agent

def create_eda_agent(llm):
    """Agent responsible for exploratory data analysis"""
    return Agent(
        role='EDA Specialist',
        goal='Perform comprehensive exploratory data analysis and statistical testing',
        backstory='''You are a statistical analysis expert with deep knowledge of 
        descriptive statistics, distributions, correlations, and hypothesis testing. 
        You can identify patterns, outliers, and relationships in data. You always 
        provide clear statistical interpretations.''',
        llm=llm,
        verbose=True,
        allow_delegation=False
    )