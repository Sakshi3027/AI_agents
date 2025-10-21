from crewai import Agent

def create_model_selector_agent(llm):
    """Agent responsible for selecting appropriate ML algorithms"""
    return Agent(
        role='ML Model Selection Expert',
        goal='Analyze the problem and data characteristics to recommend the best machine learning algorithms',
        backstory='''You are a machine learning expert who understands the strengths 
        and weaknesses of different algorithms. You can determine whether a problem is 
        classification or regression, assess data characteristics like size, features, 
        and class balance, and recommend appropriate algorithms. You understand 
        ensemble methods, boosting, and modern ML techniques.''',
        llm=llm,
        verbose=True,
        allow_delegation=False
    )