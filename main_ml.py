from dotenv import load_dotenv
import os
from crewai import Task, Crew

# Import all agents
from agents.data_loader_agent import create_data_loader_agent
from agents.eda_agent import create_eda_agent
from agents.feature_engineer_agent import create_feature_engineer_agent
from agents.model_selector_agent import create_model_selector_agent
from agents.model_trainer_agent import create_model_trainer_agent
from agents.model_evaluator_agent import create_model_evaluator_agent
from agents.report_agent import create_report_agent

# Import utilities
from utils.data_utils import load_dataset, get_dataset_summary
from utils.ml_utils import (
    prepare_ml_data, 
    train_multiple_models, 
    create_ml_visualizations,
    save_best_model
)

# Load environment variables
load_dotenv()

print("🤖 AutoAnalyst: ML Pipeline - Phase 2")
print("=" * 70)

# Check API key
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    print("❌ OPENAI_API_KEY not found in .env file!")
    exit(1)

print(f"✅ API Key loaded: {openai_key[:10]}...")

# Configuration
LLM_MODEL = "gpt-4o-mini"
DATA_FILE = "data/healthcare_data.csv"
TARGET_COLUMN = "heart_disease"  # What we want to predict

# Step 1: Load data
print("\n📊 Step 1: Loading dataset...")
df = load_dataset(DATA_FILE)
if df is None:
    print("❌ Failed to load dataset!")
    exit(1)

summary = get_dataset_summary(df)
print(f"   - Shape: {summary['shape']}")
print(f"   - Target column: {TARGET_COLUMN}")
print(f"   - Target distribution: {df[TARGET_COLUMN].value_counts().to_dict()}")

# Step 2: Prepare ML data
print("\n🔧 Step 2: Preparing data for machine learning...")
X_train, X_test, y_train, y_test, feature_names, scaler = prepare_ml_data(df, TARGET_COLUMN)

# Step 3: Train multiple models
print("\n🤖 Step 3: Training machine learning models...")
results, trained_models = train_multiple_models(X_train, X_test, y_train, y_test)

# Step 4: Create ML visualizations
print("\n📊 Step 4: Creating ML visualizations...")
ml_viz_files = create_ml_visualizations(results, y_test, feature_names, trained_models)

# Step 5: Save best model
print("\n💾 Step 5: Saving best model...")
best_model_name, model_path = save_best_model(trained_models, results)

# Step 6: Initialize AI agents for analysis
print("\n🤖 Step 6: Initializing AI agents for ML analysis...")
data_loader = create_data_loader_agent(LLM_MODEL)
eda_specialist = create_eda_agent(LLM_MODEL)
feature_engineer = create_feature_engineer_agent(LLM_MODEL)
model_selector = create_model_selector_agent(LLM_MODEL)
model_trainer = create_model_trainer_agent(LLM_MODEL)
model_evaluator = create_model_evaluator_agent(LLM_MODEL)
report_writer = create_report_agent(LLM_MODEL)

print("   ✅ 7 agents initialized")

# Prepare comprehensive ML context
ml_context = f"""
Machine Learning Analysis Summary:

Dataset: {DATA_FILE}
Target Variable: {TARGET_COLUMN}
Problem Type: Binary Classification (Heart Disease Prediction)

Data Split:
- Training samples: {len(y_train)}
- Test samples: {len(y_test)}
- Number of features: {len(feature_names)}

Feature Names: {', '.join(feature_names[:10])}... (showing first 10)

Model Performance Results:
"""

for model_name, metrics in results.items():
    # Safe handling for roc_auc (could be None)
    roc_auc_val = metrics.get('roc_auc')
    roc_auc_str = f"{roc_auc_val:.4f}" if roc_auc_val is not None else "N/A"

    ml_context += f"""
{model_name}:
  - Accuracy: {metrics['accuracy']:.4f}
  - Precision: {metrics['precision']:.4f}
  - Recall: {metrics['recall']:.4f}
  - F1-Score: {metrics['f1']:.4f}
  - ROC-AUC: {roc_auc_str}
"""

ml_context += f"""
Best Model: {best_model_name}
Model saved at: {model_path}

Visualizations Created:
{chr(10).join(f"- {viz}" for viz in ml_viz_files)}
"""

# Step 7: Create ML Analysis Tasks
print("\n📝 Step 7: Creating ML analysis tasks...")

task1 = Task(
    description=f'''Analyze the machine learning problem setup and data preparation:
    
    {ml_context}
    
    Evaluate:
    1. Is the data preparation appropriate?
    2. Are the train-test split ratios reasonable?
    3. Were features properly encoded and scaled?
    4. Any potential data leakage concerns?
    ''',
    agent=data_loader,
    expected_output='Data preparation quality assessment'
)

task2 = Task(
    description=f'''Analyze the features used for modeling:
    
    Features: {', '.join(feature_names)}
    
    Target: {TARGET_COLUMN}
    
    Assess:
    1. Are these features relevant for predicting heart disease?
    2. Are there any obvious missing features?
    3. Should any feature engineering be applied?
    4. Recommend additional features that could improve predictions
    ''',
    agent=feature_engineer,
    expected_output='Feature engineering recommendations'
)

task3 = Task(
    description=f'''Evaluate the model selection strategy:
    
    Models trained:
    {chr(10).join(f"- {name}" for name in results.keys())}
    
    Analyze:
    1. Are these appropriate models for binary classification?
    2. Should any other algorithms be considered?
    3. What are the strengths/weaknesses of each chosen model?
    4. Recommend ensemble or stacking approaches if beneficial
    ''',
    agent=model_selector,
    expected_output='Model selection analysis and recommendations'
)

task4 = Task(
    description=f'''Analyze the training process and results:
    
    {ml_context}
    
    Evaluate:
    1. Training approach and methodology
    2. Cross-validation strategy (if any)
    3. Hyperparameter tuning recommendations
    4. Risk of overfitting or underfitting
    5. Suggestions for improving model performance
    ''',
    agent=model_trainer,
    expected_output='Training process evaluation and improvement suggestions'
)

task5 = Task(
    description=f'''Comprehensively evaluate all model performances:
    
    {ml_context}
    
    Analyze:
    1. Compare all models across different metrics
    2. Which model is truly the "best" and why?
    3. Trade-offs between precision and recall
    4. Clinical implications of false positives vs false negatives
    5. ROC curve interpretation
    6. Feature importance insights
    7. Final model recommendation with justification
    ''',
    agent=model_evaluator,
    expected_output='Comprehensive model evaluation report'
)

task6 = Task(
    description='''Create a complete machine learning project report:
    
    Include:
    1. Executive Summary
    2. Problem Statement
    3. Data Preparation Methodology
    4. Feature Engineering Insights
    5. Model Selection Rationale
    6. Training Approach
    7. Performance Comparison
    8. Best Model Recommendation
    9. Clinical Interpretation
    10. Deployment Recommendations
    11. Future Improvements
    
    Write professionally for both technical and non-technical stakeholders.
    ''',
    agent=report_writer,
    expected_output='Complete ML project report'
)

print("   ✅ 6 tasks created")

# Step 8: Create and run the ML analysis crew
print("\n🎯 Step 8: Running ML analysis crew...")
print("   (This will take 3-5 minutes...)\n")

ml_crew = Crew(
    agents=[
        data_loader,
        feature_engineer,
        model_selector,
        model_trainer,
        model_evaluator,
        report_writer
    ],
    tasks=[task1, task2, task3, task4, task5, task6],
    verbose=True
)

# Run the crew
result = ml_crew.kickoff()

# Step 9: Save the ML report
print("\n💾 Step 9: Saving ML analysis report...")
ml_report_path = "outputs/ml_analysis_report.md"
with open(ml_report_path, 'w') as f:
    f.write("# Machine Learning Analysis Report\n\n")
    f.write("## Model Performance Summary\n\n")
    for model_name, metrics in results.items():
        roc_auc_val = metrics.get('roc_auc')
        roc_auc_str = f"{roc_auc_val:.4f}" if roc_auc_val is not None else "N/A"

        f.write(f"### {model_name}\n")
        f.write(f"- Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"- Precision: {metrics['precision']:.4f}\n")
        f.write(f"- Recall: {metrics['recall']:.4f}\n")
        f.write(f"- F1-Score: {metrics['f1']:.4f}\n")
        f.write(f"- ROC-AUC: {roc_auc_str}\n\n")
    
    f.write(f"## Best Model: {best_model_name}\n\n")
    f.write(f"Model saved at: `{model_path}`\n\n")
    f.write("---\n\n")
    f.write("## AI Agent Analysis\n\n")
    f.write(str(result))

print(f"✅ ML Report saved to: {ml_report_path}")

print("\n" + "=" * 70)
print("🎉 Phase 2 Complete - Machine Learning Pipeline!")
print(f"📊 ML Visualizations: outputs/")
print(f"📄 ML Report: {ml_report_path}")
print(f"🤖 Best Model: {model_path}")
print("=" * 70)
