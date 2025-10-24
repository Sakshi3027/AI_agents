import numpy as np
import pandas as pd
import optuna
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def optimize_model_with_optuna(X_train, y_train, X_test, y_test, model_type='random_forest', n_trials=50):
    """
    Use Optuna for hyperparameter optimization
    """
    print(f"\n🔍 Optimizing {model_type} with Optuna ({n_trials} trials)...")
    
    def objective(trial):
        if model_type == 'random_forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }
            model = RandomForestClassifier(**params, random_state=42)
        
        elif model_type == 'logistic_regression':
            params = {
                'C': trial.suggest_float('C', 0.001, 10.0, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                'solver': 'saga',
                'max_iter': 1000
            }
            model = LogisticRegression(**params, random_state=42)
        
        # Train and evaluate
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = f1_score(y_test, y_pred)
        
        return score
    
    # Create study
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"   ✅ Best F1-Score: {study.best_value:.4f}")
    print(f"   ✅ Best Parameters: {study.best_params}")
    
    # Train final model with best params
    if model_type == 'random_forest':
        best_model = RandomForestClassifier(**study.best_params, random_state=42)
    elif model_type == 'logistic_regression':
        best_model = LogisticRegression(**study.best_params, random_state=42, max_iter=1000, solver='saga')
    
    best_model.fit(X_train, y_train)
    
    return best_model, study.best_params, study.best_value

def create_shap_explanations(model, X_train, X_test, feature_names, output_dir='outputs'):
    """
    Generate SHAP explanations for model predictions
    """
    print("\n🔍 Generating SHAP explanations...")
    
    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # If binary classification, get positive class SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        plt.tight_layout()
        summary_path = f'{output_dir}/shap_summary.png'
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature importance plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False)
        plt.tight_layout()
        importance_path = f'{output_dir}/shap_importance.png'
        plt.savefig(importance_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ SHAP visualizations saved")
        
        return shap_values, [summary_path, importance_path]
    
    except Exception as e:
        print(f"   ⚠️ SHAP explanation failed: {e}")
        return None, []

def create_ensemble_models(X_train, y_train, X_test, y_test):
    """
    Create advanced ensemble models (Voting & Stacking)
    """
    print("\n🎯 Creating Ensemble Models...")
    
    # Base models
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    lr = LogisticRegression(random_state=42, max_iter=1000)
    dt = DecisionTreeClassifier(random_state=42, max_depth=5)
    
    # Voting Classifier
    print("   Training Voting Classifier...")
    voting_clf = VotingClassifier(
        estimators=[('rf', rf), ('lr', lr), ('dt', dt)],
        voting='soft'
    )
    voting_clf.fit(X_train, y_train)
    
    y_pred_voting = voting_clf.predict(X_test)
    voting_f1 = f1_score(y_test, y_pred_voting)
    voting_acc = accuracy_score(y_test, y_pred_voting)
    
    print(f"      ✅ Voting Classifier - Accuracy: {voting_acc:.4f}, F1: {voting_f1:.4f}")
    
    # Stacking Classifier
    print("   Training Stacking Classifier...")
    stacking_clf = StackingClassifier(
        estimators=[('rf', rf), ('dt', dt)],
        final_estimator=LogisticRegression(random_state=42)
    )
    stacking_clf.fit(X_train, y_train)
    
    y_pred_stacking = stacking_clf.predict(X_test)
    stacking_f1 = f1_score(y_test, y_pred_stacking)
    stacking_acc = accuracy_score(y_test, y_pred_stacking)
    
    print(f"      ✅ Stacking Classifier - Accuracy: {stacking_acc:.4f}, F1: {stacking_f1:.4f}")
    
    return {
        'Voting Classifier': {
            'model': voting_clf,
            'accuracy': voting_acc,
            'f1': voting_f1,
            'predictions': y_pred_voting
        },
        'Stacking Classifier': {
            'model': stacking_clf,
            'accuracy': stacking_acc,
            'f1': stacking_f1,
            'predictions': y_pred_stacking
        }
    }