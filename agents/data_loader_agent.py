from crewai import Agent

def create_data_loader_agent(llm):
    """Agent responsible for loading and validating datasets"""
    return Agent(
        role='Data Loader Specialist',
        goal='Load datasets from various formats and validate data quality',
        backstory='''You are an expert in data loading and validation. 
        You can read CSV, Excel, and JSON files. You check for data quality issues 
        like missing values, duplicates, and data types. You provide a clear 
        summary of the dataset structure.''',
        llm=llm,
        verbose=True,
        allow_delegation=False
    )
