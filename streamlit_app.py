import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_utils import get_dataset_summary, perform_eda, create_visualizations
from utils.ml_utils import prepare_ml_data, train_multiple_models, create_ml_visualizations

# Page config
st.set_page_config(
    page_title="AutoAnalyst - AI Data Science Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        transition: all 0.3s;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'ml_results' not in st.session_state:
    st.session_state.ml_results = None

# Header
st.markdown('<h1 class="main-header">🤖 AutoAnalyst</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Data Science Assistant with Multi-Agent Intelligence</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=100)
    st.title("Navigation")
    
    page = st.radio(
        "Choose a page:",
        ["🏠 Home", "�� Data Analysis", "🤖 ML Pipeline", "📈 Results", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 📁 Upload Your Data")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload your dataset for analysis"
    )
    
    if uploaded_file is not None:
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded: {uploaded_file.name}")
            st.info(f"Shape: {st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} columns")
        except Exception as e:
            st.error(f"Error loading file: {e}")
    
    # Sample data option
    st.markdown("---")
    if st.button("📂 Load Sample Data"):
        if os.path.exists("data/healthcare_data.csv"):
            st.session_state.df = pd.read_csv("data/healthcare_data.csv")
            st.success("✅ Sample healthcare data loaded!")
            st.rerun()
        else:
            st.error("Sample data not found!")
    
    st.markdown("---")
    st.markdown("""
    ### 🎯 Features
    - Upload any CSV dataset
    - Automated EDA
    - ML model training
    - Interactive visualizations
    - Downloadable reports
    """)

# Main content based on page selection
if page == "🏠 Home":
    st.header("Welcome to AutoAnalyst! 👋")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Smart Analysis</h3>
            <p>Automated exploratory data analysis with AI-powered insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 ML Pipeline</h3>
            <p>Train multiple models and compare performance automatically</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📈 Visualizations</h3>
            <p>Beautiful, interactive charts and comprehensive reports</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("🚀 Getting Started")
    
    st.markdown("""
    1. **Upload your dataset** using the sidebar (CSV format)
    2. **Navigate to Data Analysis** to explore your data
    3. **Go to ML Pipeline** to train prediction models
    4. **View Results** to see comprehensive analysis
    
    Or click "Load Sample Data" to try it with our healthcare dataset!
    """)
    
    st.markdown("---")
    
    # Show architecture
    st.subheader("🏗️ System Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Phase 1: Data Analysis**
        - Data Loader Agent
        - EDA Specialist Agent
        - Visualization Expert Agent
        - Insights Analyst Agent
        - Report Writer Agent
        """)
    
    with col2:
        st.markdown("""
        **Phase 2: ML Pipeline**
        - Feature Engineer Agent
        - Model Selector Agent
        - Model Trainer Agent
        - Model Evaluator Agent
        """)

elif page == "📊 Data Analysis":
    st.header("📊 Data Analysis")
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first using the sidebar!")
        st.info("💡 Or click 'Load Sample Data' to try with example data")
    else:
        df = st.session_state.df
        
        # Dataset overview
        st.subheader("📋 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Duplicates", df.duplicated().sum())
        
        # Data preview
        st.subheader("👀 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Column info
        st.subheader("📊 Column Information")
        
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null': df.count().values,
            'Missing': df.isnull().sum().values,
            'Unique': df.nunique().values
        })
        
        st.dataframe(col_info, use_container_width=True)
        
        # Statistical summary
        st.subheader("📈 Statistical Summary")
        
        tab1, tab2 = st.tabs(["Numerical Columns", "Categorical Columns"])
        
        with tab1:
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(numerical_cols) > 0:
                st.dataframe(df[numerical_cols].describe(), use_container_width=True)
            else:
                st.info("No numerical columns found")
        
        with tab2:
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                for col in categorical_cols:
                    with st.expander(f"📊 {col}"):
                        value_counts = df[col].value_counts()
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            fig = px.bar(
                                x=value_counts.index,
                                y=value_counts.values,
                                labels={'x': col, 'y': 'Count'},
                                title=f"Distribution of {col}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.dataframe(value_counts.to_frame('Count'))
            else:
                st.info("No categorical columns found")
        
        # Interactive visualizations
        st.subheader("📊 Interactive Visualizations")
        
        viz_type = st.selectbox(
            "Select visualization type",
            ["Distribution Plot", "Correlation Heatmap", "Scatter Plot", "Box Plot"]
        )
        
        if viz_type == "Distribution Plot":
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(numerical_cols) > 0:
                col = st.selectbox("Select column", numerical_cols)
                fig = px.histogram(df, x=col, title=f"Distribution of {col}", marginal="box")
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Correlation Heatmap":
            numerical_df = df.select_dtypes(include=['int64', 'float64'])
            if len(numerical_df.columns) > 1:
                corr = numerical_df.corr()
                fig = px.imshow(
                    corr,
                    text_auto=True,
                    aspect="auto",
                    title="Correlation Heatmap",
                    color_continuous_scale="RdBu_r"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 2 numerical columns for correlation")
        
        elif viz_type == "Scatter Plot":
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(numerical_cols) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("X-axis", numerical_cols)
                with col2:
                    y_col = st.selectbox("Y-axis", numerical_cols, index=1 if len(numerical_cols) > 1 else 0)
                
                fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Box Plot":
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(numerical_cols) > 0:
                col = st.selectbox("Select column", numerical_cols)
                fig = px.box(df, y=col, title=f"Box Plot of {col}")
                st.plotly_chart(fig, use_container_width=True)

elif page == "🤖 ML Pipeline":
    st.header("🤖 Machine Learning Pipeline")
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first!")
    else:
        df = st.session_state.df
        
        st.subheader("🎯 Configure ML Pipeline")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Select target column
            target_col = st.selectbox(
                "Select Target Column (what to predict)",
                df.columns,
                help="Choose the column you want to predict"
            )
        
        with col2:
            # Problem type
            problem_type = st.selectbox(
                "Problem Type",
                ["Classification", "Regression"],
                help="Classification for categories, Regression for continuous values"
            )
        
        # Feature selection
        st.subheader("🔧 Feature Selection")
        
        available_features = [col for col in df.columns if col != target_col and col != 'patient_id']
        selected_features = st.multiselect(
            "Select features to use (leave empty to use all)",
            available_features,
            default=available_features
        )
        
        if not selected_features:
            selected_features = available_features
        
        # Train button
        st.markdown("---")
        
        if st.button("🚀 Train Models", type="primary"):
            with st.spinner("Training models... This may take a few minutes..."):
                try:
                    # Prepare data
                    st.info("📊 Preparing data...")
                    X_train, X_test, y_train, y_test, feature_names, scaler = prepare_ml_data(
                        df[selected_features + [target_col]],
                        target_col
                    )
                    
                    # Train models
                    st.info("🤖 Training multiple models...")
                    results, trained_models = train_multiple_models(
                        X_train, X_test, y_train, y_test
                    )
                    
                    # Store results
                    st.session_state.ml_results = {
                        'results': results,
                        'models': trained_models,
                        'feature_names': feature_names,
                        'y_test': y_test,
                        'target_col': target_col
                    }
                    
                    st.success("✅ Models trained successfully!")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"Error during training: {e}")
                    st.exception(e)
        
        # Display results if available
        if st.session_state.ml_results is not None:
            st.markdown("---")
            st.subheader("📊 Model Performance")
            
            results = st.session_state.ml_results['results']
            
            # Create comparison dataframe
            comparison_df = pd.DataFrame({
                'Model': list(results.keys()),
                'Accuracy': [r['accuracy'] for r in results.values()],
                'Precision': [r['precision'] for r in results.values()],
                'Recall': [r['recall'] for r in results.values()],
                'F1-Score': [r['f1'] for r in results.values()],
            })
            
            st.dataframe(
                comparison_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score']),
                use_container_width=True
            )
            
            # Best model
            best_model = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']
            st.success(f"🏆 Best Model: **{best_model}**")

elif page == "📈 Results":
    st.header("📈 Results & Insights")
    
    if st.session_state.ml_results is None:
        st.warning("⚠️ Please train models first in the ML Pipeline page!")
    else:
        results = st.session_state.ml_results['results']
        feature_names = st.session_state.ml_results['feature_names']
        models = st.session_state.ml_results['models']
        
        # Model comparison chart
        st.subheader("📊 Model Comparison")
        
        comparison_data = []
        for model_name, metrics in results.items():
            for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
                comparison_data.append({
                    'Model': model_name,
                    'Metric': metric_name.capitalize(),
                    'Score': metrics[metric_name]
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        fig = px.bar(
            comparison_df,
            x='Model',
            y='Score',
            color='Metric',
            barmode='group',
            title='Model Performance Comparison'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Confusion matrices
        st.subheader("🎯 Confusion Matrices")
        
        cols = st.columns(2)
        for idx, (model_name, result) in enumerate(results.items()):
            with cols[idx % 2]:
                cm = result['confusion_matrix']
                fig = px.imshow(
                    cm,
                    text_auto=True,
                    aspect="auto",
                    title=f"{model_name}",
                    labels=dict(x="Predicted", y="Actual"),
                    color_continuous_scale="Blues"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # ROC Curves
        st.subheader("📈 ROC Curves")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray')
        ))
        
        from sklearn.metrics import roc_curve
        
        for model_name, result in results.items():
            if result['probabilities'] is not None:
                fpr, tpr, _ = roc_curve(st.session_state.ml_results['y_test'], result['probabilities'])
                fig.add_trace(go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode='lines',
                    name=f"{model_name} (AUC = {result['roc_auc']:.3f})"
                ))
        
        fig.update_layout(
            title='ROC Curves - Model Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=800,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        if 'Random Forest' in models:
            st.subheader("🔍 Feature Importance")
            
            rf_model = models['Random Forest']
            if hasattr(rf_model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': rf_model.feature_importances_
                }).sort_values('Importance', ascending=False).head(15)
                
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Top 15 Feature Importances (Random Forest)'
                )
                st.plotly_chart(fig, use_container_width=True)

elif page == "ℹ️ About":
    st.header("ℹ️ About AutoAnalyst")
    
    st.markdown("""
    ### 🤖 What is AutoAnalyst?
    
    AutoAnalyst is an intelligent, multi-agent AI system that automates the entire data science workflow.
    Built using CrewAI and powered by GPT-4, it features specialized AI agents that collaborate to:
    
    - 📊 Perform comprehensive exploratory data analysis
    - 🤖 Train and compare multiple machine learning models
    - 📈 Generate interactive visualizations
    - 💡 Extract actionable insights
    - 📄 Create professional reports
    
    ### 🏗️ Architecture
    
    **Phase 1: Data Analysis Pipeline**
    - Data Loader Agent
    - EDA Specialist Agent
    - Visualization Expert Agent
    - Insights Analyst Agent
    - Report Writer Agent
    
    **Phase 2: ML Pipeline**
    - Feature Engineer Agent
    - Model Selector Agent
    - Model Trainer Agent
    - Model Evaluator Agent
    
    ### 🛠️ Tech Stack
    
    - **Frontend**: Streamlit
    - **AI Framework**: CrewAI
    - **LLM**: OpenAI GPT-4o-mini
    - **ML**: Scikit-learn, XGBoost, LightGBM
    - **Visualization**: Plotly, Matplotlib, Seaborn
    - **Data Processing**: Pandas, NumPy
    
    ### 👨‍💻 Developer
    
    **Sakshi**
    - GitHub: [@Sakshi3027](https://github.com/Sakshi3027)
    - Project: [AI_agents](https://github.com/Sakshi3027/AI_agents)
    
    ### 📝 License
    
    MIT License - Feel free to use and modify!
    
    ---
    
    *Built using CrewAI and Streamlit*
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>AutoAnalyst v2.0 | Built with CrewAI × Streamlit × GPT-4</p>
    <p>Made by <a href='https://github.com/Sakshi3027'>Sakshi</a></p>
</div>
""", unsafe_allow_html=True)
