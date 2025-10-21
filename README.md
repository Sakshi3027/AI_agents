#  AutoAnalyst: Multi-Agent Data Science Assistant

[Status](https://img.shields.io/badge/status-active-success.svg)
[Python](https://img.shields.io/badge/python-3.8+-blue.svg)
[License](https://img.shields.io/badge/license-MIT-blue.svg)
An intelligent multi-agent system powered by GPT-4 and CrewAI that autonomously performs end-to-end data analysis.

## 🎯 Features

### Phase 1: Data Analysis Pipeline 
- **5 Specialized AI Agents** working collaboratively
- **Automated Exploratory Data Analysis** (EDA)
- **Statistical Analysis** with correlation detection
- **Automated Visualization** generation
- **Natural Language Insights** extraction
- **Professional Report** generation

### Phase 2: Machine Learning Pipeline ✅ NEW!
- **4 ML-Specialized Agents** for intelligent modeling
- **Automated Feature Engineering** and selection
- **Multi-Algorithm Training** (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
- **Comprehensive Model Evaluation** with multiple metrics
- **Automated Model Comparison** and selection
- **Feature Importance Analysis**
- **ROC Curves & Confusion Matrices**
- **Best Model Auto-Selection** and saving

##  Demo

###  Sample Visualizations

The system automatically generates comprehensive visualizations:

#### Distribution Analysis
![Distribution Plots](screenshots/distributions.png)
*Automated distribution analysis for all numerical variables*

#### Correlation Analysis
![Correlation Heatmap](screenshots/correlation_heatmap.png)
*Intelligent correlation detection and visualization*

#### Categorical Analysis
![Categorical Distributions](screenshots/categorical_distributions.png)
*Automated analysis of categorical variables*

## 🤖 Phase 2: Machine Learning Results

### Multi-Model Training & Comparison

![Model Comparison](screenshots/model_comparison.png)
*Automated training and comparison of 4 ML algorithms*

### Confusion Matrices

![Confusion Matrices](screenshots/confusion_matrices.png)
*Detailed classification performance for all models*

### ROC Curves

![ROC Curves](screenshots/roc_curves.png)
*ROC-AUC analysis for model selection*

### Feature Importance

![Feature Importance](screenshots/feature_importance.png)
*Top 15 most predictive features identified automatically*

### ML Performance Summary

The system automatically trained and evaluated 4 models:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~0.75 | ~0.72 | ~0.68 | ~0.67 | ~0.82 |
| Decision Tree | ~0.72 | ~0.69 | ~0.64 | ~0.64 | ~0.78 |
| **Random Forest** ⭐ | **~0.81** | **~0.78** | **~0.75** | **~0.75** | **~0.88** |
| Gradient Boosting | ~0.79 | ~0.76 | ~0.72 | ~0.72 | ~0.86 |

**Best Model:** Random Forest (Auto-selected based on F1-Score)

### Key ML Insights Generated

> **Model Selection:** "Random Forest achieved the best balance between precision and recall with F1-Score of 0.75, making it optimal for heart disease prediction"

> **Feature Importance:** "Age, blood pressure (systolic & diastolic), and smoking status are the top predictors of heart disease"

> **Clinical Recommendation:** "The model achieves 81% accuracy with strong recall (75%), minimizing false negatives which is critical in healthcare applications"

###  Agent Workflow

The system executes 5 specialized agents sequentially:

1. **Data Loader Specialist** → Validates data quality
2. **EDA Specialist** → Performs statistical analysis  
3. **Visualization Expert** → Interprets charts and patterns
4. **Insights Analyst** → Generates actionable recommendations
5. **Report Writer** → Creates professional documentation

Each agent produces detailed output that feeds into the next agent's analysis.

### Sample Report Output

The system generates a comprehensive markdown report including:

- **Executive Summary**: High-level overview of findings
- **Data Quality Report**: Completeness, missing values, duplicates
- **Statistical Analysis**: Descriptive stats, correlations, distributions
- **Visualization Insights**: Interpretation of patterns and trends
- **Key Recommendations**: Actionable insights for stakeholders
- **Conclusions**: Summary and next steps

### ⚡ Performance

**Example Analysis Time:**
- Dataset: 500 rows × 11 columns
- Analysis Duration: ~2-3 minutes
- Outputs: 3 visualizations + comprehensive report

### Sample Insights Generated

> **Finding:** "Strong positive correlation (0.384) between age and systolic blood pressure suggests age-targeted interventions for hypertension management."

> **Health Risk:** "36.6% of patients diagnosed with heart disease. Smoking shows significant association (χ² = 25.81, p<0.001), requiring targeted cessation programs."

> **Recommendation:** "Implement preventative health screenings for blood pressure and BMI in middle-aged patients to enable early detection."
## Architecture
```
┌─────────────────────────────────────────┐
│     Data Loader Agent                   │
│  (Data Quality & Validation)            │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│     EDA Specialist Agent                │
│  (Statistical Analysis)                 │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│     Visualization Expert Agent          │
│  (Chart Generation & Interpretation)    │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│     Insights Analyst Agent              │
│  (Pattern Recognition & Recommendations)│
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│     Report Writer Agent                 │
│  (Professional Documentation)           │
└─────────────────────────────────────────┘
```

## Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Sakshi3027/AutoAnalyst.git
cd AutoAnalyst
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Mac/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

## 🚀 Usage

### Run Phase 1: Data Analysis
```bash
python main.py
```

Generates:
- Exploratory data analysis
- Statistical insights
- Visualizations
- Comprehensive report

### Run Phase 2: Machine Learning Pipeline
```bash
python main_ml.py
```

Generates:
- Trained ML models
- Model comparison charts
- Confusion matrices
- ROC curves
- Feature importance analysis
- ML analysis report
- Saved best model (`.pkl`)

### Use the Saved Model
```python
import joblib
import pandas as pd

# Load the best model
model = joblib.load('outputs/best_model.pkl')

# Make predictions on new data
new_patient = pd.DataFrame({...})  # Your patient data
prediction = model.predict(new_patient)
probability = model.predict_proba(new_patient)
```

## Output

- **Visualizations**: `outputs/*.png`
- **Analysis Report**: `outputs/analysis_report.md`
![Complete Analysis](screenshots/analysis_collage.png)

## 🛠️ Tech Stack

- **CrewAI**: Multi-agent orchestration
- **OpenAI GPT-4o-mini**: Language model
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Visualization
- **Scikit-learn**: Machine learning
- **XGBoost & LightGBM**: Advanced ML algorithms
- **SHAP**: Model interpretability
- **Joblib**: Model persistence
- **Python 3.12**: Core language

## 📁 Project Structure
```
AI_agents/
├── agents/              # AI agent definitions
│   ├── data_loader_agent.py
│   ├── eda_agent.py
│   ├── visualization_agent.py
│   ├── insight_agent.py
│   ├── report_agent.py
│   ├── feature_engineer_agent.py      # ✨ Phase 2
│   ├── model_selector_agent.py        # ✨ Phase 2
│   ├── model_trainer_agent.py         # ✨ Phase 2
│   └── model_evaluator_agent.py       # ✨ Phase 2
├── utils/               # Helper functions
│   ├── data_utils.py
│   └── ml_utils.py                    # ✨ Phase 2
├── data/               # Sample datasets
├── outputs/            # Generated reports & visualizations
│   ├── *.png          # Visualizations
│   ├── *.md           # Analysis reports
│   └── best_model.pkl # Trained ML model ✨
├── screenshots/        # README images
├── main.py             # Phase 1: Data analysis pipeline
├── main_ml.py          # Phase 2: ML pipeline ✨
├── .env.example        # Environment template
└── requirements.txt    # Dependencies
```

##  Use Cases

- Healthcare data analysis
- Financial report generation
- Marketing analytics
- Research data exploration
- Educational projects

## 🔮 Future Enhancements

- [x] ~~Machine learning model training~~ ✅ COMPLETED (Phase 2)
- [ ] Interactive Streamlit dashboard (Phase 3)
- [ ] Web scraping for research papers
- [ ] Time series forecasting
- [ ] NLP for text analysis
- [ ] Deep learning models (Neural Networks)
- [ ] Model explainability with SHAP values
- [ ] AutoML with hyperparameter optimization
- [ ] Model deployment with FastAPI
- [ ] Docker containerization

## License

MIT License

##  Author

**SAKSHI**
- GitHub: [https://github.com/Sakshi3027]
Data Science Student | AI Enthusiast | Building Intelligent Systems


##  Acknowledgments

Built with CrewAI and powered by OpenAI GPT-4o-mini