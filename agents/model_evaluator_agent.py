from crewai import Agent

def create_model_evaluator_agent(llm):
    """Agent responsible for evaluating model performance"""
    return Agent(
        role='Model Evaluation Expert',
        goal='Comprehensively evaluate model performance using multiple metrics and provide interpretations',
        backstory='''You are an expert in model evaluation and interpretation. You 
        understand various metrics like accuracy, precision, recall, F1-score, ROC-AUC 
        for classification, and MAE, RMSE, R² for regression. You can interpret 
        confusion matrices, ROC curves, and feature importance. You provide clear 
        recommendations on model selection and improvements.''',
        llm=llm,
        verbose=True,
        allow_delegation=False
    )