from crewai import Agent

def create_insight_agent(llm):
    """Agent responsible for generating insights from analysis"""
    return Agent(
        role='Data Insights Analyst',
        goal='Extract meaningful insights and actionable recommendations from data analysis',
        backstory='''You are a senior data scientist who excels at finding hidden 
        patterns and translating complex statistical findings into clear business 
        insights. You provide actionable recommendations based on data evidence.''',
        llm=llm,
        verbose=True,
        allow_delegation=False
    )