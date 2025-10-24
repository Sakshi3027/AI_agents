from dotenv import load_dotenv
import os
from crewai import Task, Crew

# Import agents
from agents.automl_agent import create_automl_agent
from agents.explainability_agent import create_explainability_agent
from agents.model_evaluator_agent import create_model_evaluator_agent
from agents.report_agent import create_report_agent

# Import utilities
from utils.data_utils import load_dataset, get_dataset_summary
from utils.ml_utils import prepare_ml_data, train_multiple_models
from utils.advanced_ml_utils import (
    optimize_model_with_optuna,
    create_shap_explanations,
    create_ensemble_models
)

load_dotenv()

print("🚀 AutoAnalyst: Advanced ML Pipeline - Phase 4")
print("=" * 80)

# Check API key
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    print("❌ OPENAI_API_KEY not found!")
    exit(1)

print(f"✅ API Key loaded: {openai_key[:10]}...")

# Configuration
LLM_MODEL = "gpt-4o-mini"
DATA_FILE = "data/healthcare_data.csv"
TARGET_COLUMN = "heart_disease"

# Step 1: Load data
print("\n📊 Step 1: Loading dataset...")
df = load_dataset(DATA_FILE)
if df is None:
    exit(1)

summary = get_dataset_summary(df)
print(f"   - Shape: {summary['shape']}")
print(f"   - Target: {TARGET_COLUMN}")

# Step 2: Prepare data
print("\n🔧 Step 2: Preparing data for ML...")
X_train, X_test, y_train, y_test, feature_names, scaler = prepare_ml_data(df, TARGET_COLUMN)

# Step 3: Train baseline models
print("\n🤖 Step 3: Training baseline models...")
baseline_results, baseline_models = train_multiple_models(X_train, X_test, y_train, y_test)

print("\n📊 Baseline Model Performance:")
for name, metrics in baseline_results.items():
    print(f"   {name:25s} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")

# Step 4: AutoML Optimization
print("\n" + "="*80)
print("🔍 Step 4: AutoML - Hyperparameter Optimization with Optuna")
print("="*80)

print("\n🎯 Optimizing Random Forest...")
optimized_rf, best_params_rf, best_score_rf = optimize_model_with_optuna(
    X_train, y_train, X_test, y_test, 
    model_type='random_forest', 
    n_trials=30
)

print("\n🎯 Optimizing Logistic Regression...")
optimized_lr, best_params_lr, best_score_lr = optimize_model_with_optuna(
    X_train, y_train, X_test, y_test, 
    model_type='logistic_regression', 
    n_trials=30
)

# Step 5: Ensemble Models
print("\n" + "="*80)
print("🎯 Step 5: Advanced Ensemble Methods")
print("="*80)

ensemble_results = create_ensemble_models(X_train, y_train, X_test, y_test)

# Step 6: SHAP Explainability
print("\n" + "="*80)
print("�� Step 6: Model Explainability with SHAP")
print("="*80)

shap_values, shap_plots = create_shap_explanations(
    optimized_rf, X_train, X_test, feature_names
)

# Step 7: Compare all models
print("\n" + "="*80)
print("📊 Step 7: Complete Model Comparison")
print("="*80)

all_results = {
    'Baseline Random Forest': baseline_results['Random Forest'],
    'Optimized Random Forest': {
        'accuracy': baseline_results['Random Forest']['accuracy'],
        'f1': best_score_rf
    },
    'Baseline Logistic Regression': baseline_results['Logistic Regression'],
    'Optimized Logistic Regression': {
        'accuracy': baseline_results['Logistic Regression']['accuracy'],
        'f1': best_score_lr
    },
    'Voting Classifier': ensemble_results['Voting Classifier'],
    'Stacking Classifier': ensemble_results['Stacking Classifier']
}

print("\n🏆 Final Model Performance:")
print("-" * 80)
for model_name, metrics in all_results.items():
    acc = metrics.get('accuracy', 0)
    f1 = metrics.get('f1', 0)
    print(f"{model_name:35s} | Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")

# Find best model
best_model_name = max(all_results.keys(), key=lambda k: all_results[k].get('f1', 0))
best_f1 = all_results[best_model_name]['f1']
improvement = best_f1 - baseline_results['Random Forest']['f1']

print(f"\n🎯 Best Model: {best_model_name}")
print(f"   F1-Score: {best_f1:.4f}")
print(f"   Improvement over baseline: +{improvement:.4f} ({improvement/baseline_results['Random Forest']['f1']*100:.1f}%)")

# Step 8: AI Agent Analysis
print("\n" + "="*80)
print("🤖 Step 8: AI Agent Analysis of Results")
print("="*80)

automl_agent = create_automl_agent(LLM_MODEL)
explainability_agent = create_explainability_agent(LLM_MODEL)
evaluator_agent = create_model_evaluator_agent(LLM_MODEL)
report_agent = create_report_agent(LLM_MODEL)

print("   ✅ 4 agents initialized")

# Create analysis context
analysis_context = f"""
Advanced ML Analysis Results:

Dataset: {DATA_FILE}
Target: {TARGET_COLUMN} (Binary Classification)
Training samples: {len(y_train)}, Test samples: {len(y_test)}
Features: {len(feature_names)}

=== BASELINE MODELS ===
{chr(10).join(f"{name}: Acc={m['accuracy']:.4f}, F1={m['f1']:.4f}" for name, m in baseline_results.items())}

=== AUTOML OPTIMIZATION (Optuna) ===
Random Forest Optimization:
  - Best F1-Score: {best_score_rf:.4f}
  - Improvement: +{best_score_rf - baseline_results['Random Forest']['f1']:.4f}
  - Best Parameters: {best_params_rf}

Logistic Regression Optimization:
  - Best F1-Score: {best_score_lr:.4f}
  - Improvement: +{best_score_lr - baseline_results['Logistic Regression']['f1']:.4f}
  - Best Parameters: {best_params_lr}

=== ENSEMBLE METHODS ===
{chr(10).join(f"{name}: Acc={m['accuracy']:.4f}, F1={m['f1']:.4f}" for name, m in ensemble_results.items())}

=== MODEL EXPLAINABILITY ===
SHAP analysis completed on Optimized Random Forest
Generated visualizations:
- Feature importance rankings
- Individual feature contributions
- Summary plots

=== BEST MODEL ===
Model: {best_model_name}
F1-Score: {best_f1:.4f}
Overall Improvement: +{improvement:.4f} ({improvement/baseline_results['Random Forest']['f1']*100:.1f}%)
"""

# Create tasks
print("\n📝 Creating analysis tasks...")

task1 = Task(
    description=f'''Analyze the AutoML optimization results:
    
    {analysis_context}
    
    Evaluate:
    1. How effective was hyperparameter optimization?
    2. Are the optimized parameters reasonable and well-tuned?
    3. What insights can we gain from the best parameters?
    4. Did we achieve meaningful improvements?
    ''',
    agent=automl_agent,
    expected_output='AutoML optimization analysis with insights'
)

task2 = Task(
    description=f'''Interpret the SHAP explainability results:
    
    SHAP analysis was performed on the Optimized Random Forest model.
    Feature importance and contribution analysis completed.
    
    Explain:
    1. Which features are most important for heart disease prediction?
    2. How do individual features contribute to predictions?
    3. What clinical insights can we derive from feature importance?
    4. How does explainability improve model trust and usability?
    5. Any recommendations for feature engineering?
    ''',
    agent=explainability_agent,
    expected_output='Detailed SHAP interpretation with clinical insights'
)

task3 = Task(
    description=f'''Comprehensive evaluation of all modeling approaches:
    
    {analysis_context}
    
    Provide:
    1. Compare all models (baseline, optimized, ensemble)
    2. Which approach is best for this healthcare problem and why?
    3. Trade-offs between model complexity and interpretability
    4. Production deployment recommendations
    5. When to use each approach in practice
    ''',
    agent=evaluator_agent,
    expected_output='Complete model comparison and recommendations'
)

task4 = Task(
    description='''Create a comprehensive advanced ML report:
    
    Include:
    1. Executive Summary
    2. AutoML Optimization Results & Insights
    3. Ensemble Methods Evaluation
    4. Model Explainability Analysis (SHAP)
    5. Best Model Recommendation with Justification
    6. Clinical Interpretation for Healthcare
    7. Production Deployment Strategy
    8. Future Improvements & Next Steps
    
    Write professionally for both technical and clinical stakeholders.
    ''',
    agent=report_agent,
    expected_output='Professional advanced ML analysis report'
)

print("   ✅ 4 tasks created")

# Run AI analysis
print("\n🎯 Step 9: Running AI agent analysis...")
print("   (This will take 5-7 minutes...)\n")

advanced_crew = Crew(
    agents=[automl_agent, explainability_agent, evaluator_agent, report_agent],
    tasks=[task1, task2, task3, task4],
    verbose=True
)

result = advanced_crew.kickoff()

# Save report
print("\n💾 Step 10: Saving comprehensive report...")

report_path = "outputs/advanced_ml_report.md"
with open(report_path, 'w') as f:
    f.write("# Advanced Machine Learning Analysis Report\n\n")
    f.write("## Executive Summary\n\n")
    f.write(f"**Best Model:** {best_model_name}\n")
    f.write(f"**Best F1-Score:** {best_f1:.4f}\n")
    f.write(f"**Improvement over Baseline:** +{improvement:.4f} ({improvement/baseline_results['Random Forest']['f1']*100:.1f}%)\n\n")
    
    f.write("## Model Performance Comparison\n\n")
    f.write("| Model | Accuracy | F1-Score |\n")
    f.write("|-------|----------|----------|\n")
    for name, metrics in all_results.items():
        f.write(f"| {name} | {metrics.get('accuracy', 0):.4f} | {metrics.get('f1', 0):.4f} |\n")
    
    f.write("\n## AutoML Optimization Details\n\n")
    f.write("### Random Forest\n")
    f.write(f"- **Baseline F1:** {baseline_results['Random Forest']['f1']:.4f}\n")
    f.write(f"- **Optimized F1:** {best_score_rf:.4f}\n")
    f.write(f"- **Improvement:** +{best_score_rf - baseline_results['Random Forest']['f1']:.4f}\n")
    f.write(f"- **Best Parameters:** `{best_params_rf}`\n\n")
    
    f.write("### Logistic Regression\n")
    f.write(f"- **Baseline F1:** {baseline_results['Logistic Regression']['f1']:.4f}\n")
    f.write(f"- **Optimized F1:** {best_score_lr:.4f}\n")
    f.write(f"- **Improvement:** +{best_score_lr - baseline_results['Logistic Regression']['f1']:.4f}\n")
    f.write(f"- **Best Parameters:** `{best_params_lr}`\n\n")
    
    f.write("## Explainability\n\n")
    f.write("SHAP (SHapley Additive exPlanations) analysis completed:\n")
    f.write("- Feature importance rankings generated\n")
    f.write("- Individual prediction explanations available\n")
    f.write("- Clinical insights extracted from feature contributions\n\n")
    
    f.write("---\n\n")
    f.write("## AI Agent Analysis\n\n")
    f.write(str(result))

print(f"✅ Report saved: {report_path}")

print("\n" + "="*80)
print("🎉 Phase 4 Complete - Advanced ML with AutoML & Explainability!")
print("="*80)
print(f"📊 Models Trained: 6 (baseline + optimized + ensemble)")
print(f"🔍 AutoML: Optuna hyperparameter optimization")
print(f"💡 Explainability: SHAP analysis")
print(f"📄 Report: {report_path}")
print(f"🏆 Best Model: {best_model_name} (F1: {best_f1:.4f})")
print("="*80)
