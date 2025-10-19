from dotenv import load_dotenv
import os
from crewai import Task, Crew

# Import agent creators
from agents.data_loader_agent import create_data_loader_agent
from agents.eda_agent import create_eda_agent
from agents.visualization_agent import create_visualization_agent
from agents.insight_agent import create_insight_agent
from agents.report_agent import create_report_agent

# Import utilities
from utils.data_utils import load_dataset, get_dataset_summary, perform_eda, create_visualizations

# Load environment variables
load_dotenv()

print("🚀 AutoAnalyst: Multi-Agent Data Science Assistant")
print("=" * 60)

# Check API key
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    print("❌ OPENAI_API_KEY not found in .env file!")
    exit(1)

print(f"✅ API Key loaded: {openai_key[:10]}...")

# Configuration
LLM_MODEL = "gpt-4o-mini"
DATA_FILE = "data/healthcare_data.csv"

# Step 1: Load data
print("\n📊 Step 1: Loading dataset...")
df = load_dataset(DATA_FILE)
if df is None:
    print("❌ Failed to load dataset!")
    exit(1)

# Step 2: Get dataset summary
print("\n📋 Step 2: Analyzing dataset structure...")
summary = get_dataset_summary(df)
print(f"   - Shape: {summary['shape']}")
print(f"   - Missing values: {sum(summary['missing_values'].values())} total")
print(f"   - Duplicates: {summary['duplicates']}")

# Step 3: Perform EDA
print("\n🔍 Step 3: Performing exploratory data analysis...")
eda_results = perform_eda(df)
print(f"   - Analyzed {len(eda_results.get('numerical_stats', {}))} numerical columns")
print(f"   - Analyzed {len(eda_results.get('categorical_counts', {}))} categorical columns")

# Step 4: Create visualizations
print("\n📈 Step 4: Creating visualizations...")
viz_files = create_visualizations(df)
print(f"   - Created {len(viz_files)} visualization files")

# Step 5: Create AI agents
print("\n🤖 Step 5: Initializing AI agents...")
data_loader = create_data_loader_agent(LLM_MODEL)
eda_specialist = create_eda_agent(LLM_MODEL)
viz_expert = create_visualization_agent(LLM_MODEL)
insight_analyst = create_insight_agent(LLM_MODEL)
report_writer = create_report_agent(LLM_MODEL)

print("   ✅ 5 agents initialized successfully")

# Prepare context for agents
context = f"""
Dataset Analysis Summary:

File: {DATA_FILE}
Shape: {summary['shape'][0]} rows, {summary['shape'][1]} columns
Columns: {', '.join(summary['columns'])}

Missing Values:
{summary['missing_values']}

Key Statistics:
{eda_results.get('numerical_stats', {})}

Categorical Distributions:
{eda_results.get('categorical_counts', {})}

Visualizations: {len(viz_files)} plots created in outputs/ directory
"""

# Step 6: Create Tasks
print("\n📝 Step 6: Creating agent tasks...")

task1 = Task(
    description=f'''Analyze this dataset summary and provide insights about data quality:
    {context}
    
    Focus on:
    1. Data completeness and quality
    2. Variable types and distributions
    3. Potential data issues
    ''',
    agent=data_loader,
    expected_output='A comprehensive data quality report'
)

task2 = Task(
    description=f'''Perform statistical analysis on this data:
    {context}
    
    Analyze:
    1. Descriptive statistics for all variables
    2. Correlations between numerical variables
    3. Distributions and patterns
    4. Statistical significance of relationships
    ''',
    agent=eda_specialist,
    expected_output='Detailed statistical analysis report'
)

task3 = Task(
    description=f'''Interpret the visualizations created and explain patterns:
    
    Visualizations available in outputs/:
    {viz_files}
    
    Data context:
    {context}
    
    Explain what patterns, trends, and insights can be seen in the visualizations.
    ''',
    agent=viz_expert,
    expected_output='Visualization interpretation report'
)

task4 = Task(
    description='''Based on all previous analyses, generate key insights and recommendations:
    
    1. What are the most important findings?
    2. What patterns or correlations are significant?
    3. What actionable recommendations can be made?
    4. Are there any surprising or noteworthy discoveries?
    ''',
    agent=insight_analyst,
    expected_output='Key insights and recommendations'
)

task5 = Task(
    description='''Create a comprehensive final report combining all analyses:
    
    Include:
    1. Executive summary
    2. Data overview
    3. Statistical findings
    4. Visualization insights
    5. Key recommendations
    6. Conclusions
    
    Write in a professional, clear style suitable for stakeholders.
    ''',
    agent=report_writer,
    expected_output='Complete professional data analysis report'
)

print("   ✅ 5 tasks created")

# Step 7: Create and run the crew
print("\n🎯 Step 7: Assembling the crew and starting analysis...")
print("   (This may take a few minutes...)\n")

crew = Crew(
    agents=[data_loader, eda_specialist, viz_expert, insight_analyst, report_writer],
    tasks=[task1, task2, task3, task4, task5],
    verbose=True
)

# Run the crew
result = crew.kickoff()

# Step 8: Save the report
print("\n💾 Step 8: Saving final report...")
report_path = "outputs/analysis_report.md"
with open(report_path, 'w') as f:
    f.write(str(result))

print(f"✅ Report saved to: {report_path}")

print("\n" + "=" * 60)
print("🎉 Analysis Complete!")
print(f"📊 Visualizations: outputs/")
print(f"📄 Report: {report_path}")
print("=" * 60)