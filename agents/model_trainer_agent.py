from crewai import Agent

def create_model_trainer_agent(llm):
    """Agent responsible for training ML models"""
    return Agent(
        role='Model Training Specialist',
        goal='Train machine learning models with optimal hyperparameters and validate performance',
        backstory='''You are an expert in training machine learning models. You know 
        how to split data properly, apply cross-validation, handle class imbalance, 
        and tune hyperparameters. You understand overfitting, underfitting, and how 
        to achieve the best model performance through proper training techniques.''',
        llm=llm,
        verbose=True,
        allow_delegation=False
    )