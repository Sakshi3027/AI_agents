import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def prepare_ml_data(df, target_column):
    """Prepare data for machine learning"""
    print(f"\n🔧 Preparing data for ML...")
    print(f"   Target column: {target_column}")
    
    # Separate features and target
    X = df.drop(columns=[target_column, 'patient_id'])  # Remove ID column
    y = df[target_column]
    
    # Encode target if categorical
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        print(f"   Target encoded: {le.classes_}")
    
    # Handle categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print(f"   Encoding {len(categorical_cols)} categorical columns...")
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"   ✅ Data prepared:")
    print(f"      - Training set: {X_train.shape[0]} samples")
    print(f"      - Test set: {X_test.shape[0]} samples")
    print(f"      - Features: {X_train.shape[1]}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist(), scaler

def train_multiple_models(X_train, X_test, y_train, y_test):
    """Train multiple ML models and compare performance"""
    print("\n🤖 Training multiple models...")
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n   Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1': f1_score(y_test, y_pred, average='binary'),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"      ✅ Accuracy: {results[name]['accuracy']:.4f}")
        print(f"      ✅ F1-Score: {results[name]['f1']:.4f}")
    
    return results, trained_models

def create_ml_visualizations(results, y_test, feature_names, trained_models, output_dir='outputs'):
    """Create ML performance visualizations"""
    print("\n📊 Creating ML visualizations...")
    os.makedirs(output_dir, exist_ok=True)
    viz_files = []
    
    # 1. Model Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    models = list(results.keys())
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in models]
        ax.bar(x + i*width, values, width, label=metric.capitalize())
    
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    filepath = f'{output_dir}/model_comparison.png'
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    viz_files.append(filepath)
    
    # 2. Confusion Matrices
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (name, result) in enumerate(results.items()):
        cm = result['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   cbar_kws={'label': 'Count'})
        axes[idx].set_title(f'{name}\nConfusion Matrix', fontweight='bold')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    filepath = f'{output_dir}/confusion_matrices.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    viz_files.append(filepath)
    
    # 3. ROC Curves
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for name, result in results.items():
        if result['probabilities'] is not None:
            fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
            auc = result['roc_auc']
            ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    
    filepath = f'{output_dir}/roc_curves.png'
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    viz_files.append(filepath)
    
    # 4. Feature Importance (for tree-based models)
    rf_model = trained_models.get('Random Forest')
    if rf_model and hasattr(rf_model, 'feature_importances_'):
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]  # Top 15 features
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(indices)), importances[indices], color='steelblue')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
        ax.set_title('Top 15 Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        filepath = f'{output_dir}/feature_importance.png'
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        viz_files.append(filepath)
    
    print(f"   ✅ Created {len(viz_files)} ML visualization files")
    return viz_files

def save_best_model(trained_models, results, output_dir='outputs'):
    """Save the best performing model"""
    # Find best model based on F1 score
    best_model_name = max(results.keys(), key=lambda k: results[k]['f1'])
    best_model = trained_models[best_model_name]
    
    model_path = f'{output_dir}/best_model.pkl'
    joblib.dump(best_model, model_path)
    
    print(f"\n💾 Best model saved: {best_model_name}")
    print(f"   Path: {model_path}")
    print(f"   F1 Score: {results[best_model_name]['f1']:.4f}")
    
    return best_model_name, model_path