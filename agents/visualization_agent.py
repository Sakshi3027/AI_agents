from crewai import Agent

def create_visualization_agent(llm):
    """Agent responsible for creating data visualizations"""
    return Agent(
        role='Data Visualization Expert',
        goal='Create insightful and beautiful visualizations that reveal data patterns',
        backstory='''You are a data visualization expert who knows when to use 
        different chart types. You create clear, informative plots including 
        distributions, correlations, time series, and comparisons. You follow 
        best practices in data visualization.''',
        llm=llm,
        verbose=True,
        allow_delegation=False
    )