import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_dataset(filepath):
    """Load dataset from file"""
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(filepath)
        elif filepath.endswith('.json'):
            df = pd.read_json(filepath)
        else:
            raise ValueError("Unsupported file format")
        
        print(f"✅ Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def get_dataset_summary(df):
    """Get comprehensive dataset summary"""
    summary = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
    }
    return summary

def perform_eda(df):
    """Perform exploratory data analysis"""
    results = {}
    
    # Basic statistics
    results['numerical_stats'] = df.describe().to_dict()
    
    # Categorical value counts
    categorical_cols = df.select_dtypes(include=['object']).columns
    results['categorical_counts'] = {
        col: df[col].value_counts().to_dict() 
        for col in categorical_cols
    }
    
    # Correlations
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 1:
        results['correlations'] = df[numerical_cols].corr().to_dict()
    
    return results

def create_visualizations(df, output_dir='outputs'):
    """Create standard visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    viz_files = []
    
    # 1. Distribution plots for numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, col in enumerate(numerical_cols[:6]):
            df[col].hist(bins=30, ax=axes[idx], edgecolor='black')
            axes[idx].set_title(f'Distribution of {col}')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
        
        plt.tight_layout()
        filepath = f'{output_dir}/distributions.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        viz_files.append(filepath)
    
    # 2. Correlation heatmap
    if len(numerical_cols) > 1:
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=1, fmt='.2f')
        plt.title('Correlation Heatmap')
        filepath = f'{output_dir}/correlation_heatmap.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        viz_files.append(filepath)
    
    # 3. Categorical plots
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        fig, axes = plt.subplots(1, min(3, len(categorical_cols)), figsize=(15, 5))
        if len(categorical_cols) == 1:
            axes = [axes]
        
        for idx, col in enumerate(categorical_cols[:3]):
            df[col].value_counts().plot(kind='bar', ax=axes[idx], edgecolor='black')
            axes[idx].set_title(f'Distribution of {col}')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Count')
            axes[idx].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        filepath = f'{output_dir}/categorical_distributions.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        viz_files.append(filepath)
    
    print(f"✅ Created {len(viz_files)} visualizations in {output_dir}/")
    return viz_files