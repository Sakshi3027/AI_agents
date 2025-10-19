from crewai import Agent

def create_report_agent(llm):
    """Agent responsible for writing comprehensive reports"""
    return Agent(
        role='Technical Report Writer',
        goal='Create professional, well-structured data analysis reports',
        backstory='''You are an expert technical writer who creates comprehensive 
        data analysis reports. You organize findings logically, include relevant 
        statistics and visualizations, and write in clear, professional language. 
        Your reports are publication-ready.''',
        llm=llm,
        verbose=True,
        allow_delegation=False
    )