# AutoAnalyst: Multi-Agent Data Science Assistant

An intelligent multi-agent system that autonomously performs end-to-end data analysis, ML model training, and report generation — powered by CrewAI and GPT-4.

---

## What It Does

You upload any CSV file. The system automatically:
- Analyzes the data (EDA, statistics, correlations)
- Trains and compares multiple ML models
- Generates interactive visualizations
- Produces a full written report

No manual steps. No code required from the end user.

---

## System Architecture

The system is organized into 5 phases, each adding a new layer of capability:

**Phase 1 — Data Analysis Pipeline**
5 specialized AI agents run sequentially, each passing output to the next:
1. Data Loader Agent — validates data quality and structure
2. EDA Specialist Agent — performs statistical analysis
3. Visualization Expert Agent — generates and interprets charts
4. Insights Analyst Agent — identifies patterns and recommendations
5. Report Writer Agent — produces a professional markdown report

**Phase 2 — Machine Learning Pipeline**
4 ML agents handle automated modeling:
- Feature Engineer Agent — selects and transforms features
- Model Selector Agent — chooses appropriate algorithms
- Model Trainer Agent — trains Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- Model Evaluator Agent — compares models on Accuracy, Precision, Recall, F1, ROC-AUC

**Phase 3 — Streamlit Web Dashboard**
Interactive UI built with Streamlit:
- Drag-and-drop CSV upload
- Real-time data exploration with Plotly charts
- One-click ML model training with live progress
- Results dashboard with model comparison and feature importance

**Phase 4 — Advanced ML**
- AutoML with Optuna for hyperparameter tuning — achieved 37.1% F1 improvement (0.30 to 0.41)
- SHAP explainability — shows why the model made each prediction
- Ensemble methods — Voting and Stacking classifiers

**Phase 5 — Production API**
- FastAPI REST API deployed on Render.com
- Docker containerized for consistent deployment
- Swagger UI documentation at `/docs`
- Response time under 200ms
- Supports single prediction and batch CSV prediction

---

## ML Results (Healthcare Dataset)

Trained and evaluated on 500 patient records predicting heart disease:

| Model               | Accuracy | F1-Score | ROC-AUC |
|---------------------|----------|----------|---------|
| Logistic Regression | 0.75     | 0.67     | 0.82    |
| Decision Tree       | 0.72     | 0.64     | 0.78    |
| Random Forest       | 0.81     | 0.75     | 0.88    |
| Gradient Boosting   | 0.79     | 0.72     | 0.86    |

**Best Model:** Random Forest — auto-selected based on F1-Score

**Top Predictors (via SHAP):** Cholesterol, Age, Blood Pressure

**AutoML Result:** Random Forest F1 improved from 0.30 to 0.41 (+37.1%) using Optuna

---

## API Endpoints

| Method | Endpoint        | Description                        |
|--------|-----------------|------------------------------------|
| GET    | /               | Welcome message and API info       |
| GET    | /health         | Health check and model status      |
| POST   | /predict        | Single patient prediction          |
| POST   | /batch-predict  | Batch predictions from CSV         |
| GET    | /stats          | Usage statistics                   |
| GET    | /model-info     | Model details and features         |
| GET    | /docs           | Interactive Swagger UI             |

Live API: `https://autoanalyst-api.onrender.com`

---

## Tech Stack

| Layer          | Technology                                      |
|----------------|-------------------------------------------------|
| AI Agents      | CrewAI                                          |
| Language Model | OpenAI GPT-4o-mini                              |
| ML             | Scikit-learn, XGBoost, LightGBM                 |
| AutoML         | Optuna                                          |
| Explainability | SHAP                                            |
| API            | FastAPI                                         |
| Frontend       | Streamlit                                       |
| Visualization  | Plotly, Matplotlib, Seaborn                     |
| Data           | Pandas, NumPy                                   |
| Deployment     | Docker, Docker Compose, Render.com              |
| CI/CD          | GitHub Actions                                  |
| Language       | Python 3.12                                     |

---

## Screenshots

### Web Dashboard

#### Home Page
![Home](screenshots/streamlit_home.png)

#### ML Training in Progress
![Training](screenshots/streamlit_training.png)

#### Model Performance Table
![Performance](screenshots/streamlit_performance.png)

#### Results & Insights
![Results](screenshots/streamlit_results.png)

---

### Data Analysis Outputs

#### Distribution Analysis
![Distributions](screenshots/distributions.png)

#### Correlation Heatmap
![Correlation](screenshots/correlation_heatmap.png)

#### Categorical Analysis
![Categorical](screenshots/categorical_distributions.png)

---

### ML Results

#### Model Performance Comparison
![Model Comparison](screenshots/model_comparison.png)

#### Confusion Matrices
![Confusion Matrices](screenshots/confusion_matrices.png)

#### ROC Curves
![ROC Curves](screenshots/roc_curves.png)

#### Feature Importance (Random Forest)
![Feature Importance](screenshots/feature_importance.png)

#### Full Analysis Collage
![Analysis Collage](screenshots/analysis_collage.png)

---

## Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key

### Installation

```bash
git clone https://github.com/Sakshi3027/AutoAnalyst.git
cd AutoAnalyst
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

### Run the Web Dashboard (Recommended)
```bash
streamlit run streamlit_app.py
```
Opens at `http://localhost:8501`

**Steps:**
1. Click "Load Sample Data" in the sidebar
2. Go to "Data Analysis" to explore
3. Go to "ML Pipeline" and click "Train Models"
4. View "Results" for full analysis

### Run Data Analysis Only
```bash
python main.py
```
Output: `outputs/analysis_report.md`

### Run ML Pipeline Only
```bash
python main_ml.py
```
Output: `outputs/ml_analysis_report.md` and `outputs/best_model.pkl`

### Run Advanced ML (AutoML + SHAP)
```bash
python main_advanced.py
```

---

## Docker Deployment

```bash
# Build
docker build -t autoanalyst .

# Run
docker run -p 8000:8000 --env-file .env autoanalyst

# Or with docker-compose
docker-compose up -d
```

---

## Use Saved Model

```python
import joblib
import pandas as pd

model = joblib.load('outputs/best_model.pkl')

new_patient = pd.DataFrame({
    'age': [55],
    'bmi': [28.5],
    'blood_pressure_systolic': [140],
    'blood_pressure_diastolic': [90],
    'cholesterol': [220],
    'glucose': [120],
    'exercise_hours_per_week': [3],
    'gender_Male': [1],
    'smoker_Yes': [1]
})

prediction = model.predict(new_patient)
probability = model.predict_proba(new_patient)

print(f"Prediction: {'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'}")
print(f"Probability: {probability[0][1]:.2%}")
```

---

## Project Structure

```
AI_agents/
├── agents/                        # 12 AI agent definitions
│   ├── data_loader_agent.py
│   ├── eda_agent.py
│   ├── visualization_agent.py
│   ├── insight_agent.py
│   ├── report_agent.py
│   ├── feature_engineer_agent.py
│   ├── model_selector_agent.py
│   ├── model_trainer_agent.py
│   ├── model_evaluator_agent.py
│   ├── automl_agent.py
│   ├── explainability_agent.py
│   └── deep_learning_agent.py
├── utils/
│   ├── data_utils.py              # EDA and visualization helpers
│   └── ml_utils.py                # ML training and evaluation
├── data/
│   └── healthcare_data.csv        # Sample dataset (500 records)
├── outputs/                       # Generated reports and models
├── screenshots/                   # UI screenshots
├── api.py                         # FastAPI REST API
├── Dockerfile
├── docker-compose.yml
├── main.py                        # Phase 1 pipeline
├── main_ml.py                     # Phase 2 ML pipeline
├── main_advanced.py               # Phase 4 advanced ML
├── streamlit_app.py               # Phase 3 web dashboard
├── requirements.txt
└── README.md
```

---

## Key Achievements

- 12 AI agents for fully automated data science pipeline
- 37.1% F1-Score improvement through AutoML (Optuna)
- Production API deployed on Render.com with under 200ms response time
- SHAP explainability for model transparency
- Full Docker containerization
- End-to-end: raw CSV in, trained model + report out

---

## Use Cases

- Healthcare data analysis and prediction
- Financial report generation
- Marketing analytics
- Research data exploration
- Enterprise data science automation

---

## Author

**Sakshi Chavan**
- GitHub: [github.com/Sakshi3027](https://github.com/Sakshi3027)
- Live API: [autoanalyst-api.onrender.com](https://autoanalyst-api.onrender.com)

---

## License

MIT License