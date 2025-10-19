import pandas as pd
import numpy as np

# Create sample healthcare dataset
np.random.seed(42)

n_samples = 500

data = {
    'patient_id': range(1, n_samples + 1),
    'age': np.random.randint(18, 85, n_samples),
    'gender': np.random.choice(['Male', 'Female'], n_samples),
    'bmi': np.random.normal(27, 5, n_samples).round(1),
    'blood_pressure_systolic': np.random.randint(90, 180, n_samples),
    'blood_pressure_diastolic': np.random.randint(60, 120, n_samples),
    'cholesterol': np.random.randint(150, 300, n_samples),
    'glucose': np.random.randint(70, 200, n_samples),
    'smoker': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
    'exercise_hours_per_week': np.random.randint(0, 15, n_samples),
    'heart_disease': np.random.choice(['Yes', 'No'], n_samples, p=[0.25, 0.75])
}

df = pd.DataFrame(data)

# Add some realistic correlations
df.loc[df['smoker'] == 'Yes', 'heart_disease'] = np.random.choice(
    ['Yes', 'No'], 
    sum(df['smoker'] == 'Yes'), 
    p=[0.4, 0.6]
)

df.loc[df['age'] > 60, 'heart_disease'] = np.random.choice(
    ['Yes', 'No'], 
    sum(df['age'] > 60), 
    p=[0.45, 0.55]
)

# Save to CSV
df.to_csv('data/healthcare_data.csv', index=False)
print("✅ Sample dataset created: data/healthcare_data.csv")
print(f"📊 Shape: {df.shape}")
print("\n🔍 Preview:")
print(df.head())
