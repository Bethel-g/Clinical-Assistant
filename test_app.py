import pandas as pd
import joblib

artifacts = {}
required_files = [
    'preprocessor.joblib',
    'disease_model.joblib',
    'risk_model.joblib',
    'stay_model.joblib',
    'disease_label_encoder.joblib',
    'risk_label_encoder.joblib',
    'feature_names.joblib'
]
for f in required_files:
    artifacts[f] = joblib.load(f"models/{f}")

preprocessor = artifacts['preprocessor.joblib']

inputs = {
    'Age': 35, 'Gender': 'Male', 'Region': 'Addis Ababa', 'Fever': 'No',
    'Cough': 'No', 'Headache': 'No', 'Fatigue': 'No', 'Vomiting': 'No',
    'Diarrhea': 'No', 'Chest Pain': 'No', 'Shortness of Breath': 'No',
    'Dizziness': 'No', 'Temperature': 37.0, 'Heart Rate': 80, 'WBC Count': 8000,
    'Hemoglobin': 13.0, 'Malaria Test': 'Negative', 'Comorbidity': 'No',
    'Season': 'Summer', 'Weight': 70.0, 'Height': 170.0, 'BMI': 24.2,
    'Oxygen Saturation': 98.0, 'Blood Pressure Systolic': 120,
    'Blood Pressure Diastolic': 80, 'Pain Score': 0, 'Symptoms': '',
    'Free Text Symptoms': '', 'Clinical Notes': ''
}

input_df = pd.DataFrame([inputs])

print("Columns in input_df:", input_df.columns.tolist())
try:
    processed_input = preprocessor.transform(input_df)
    print("Preprocessing successful")
    
    disease_model = artifacts['disease_model.joblib']
    pred = disease_model.predict(input_df)
    print("Prediction successful:", pred)
except Exception as e:
    import traceback
    traceback.print_exc()
