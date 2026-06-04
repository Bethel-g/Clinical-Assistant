import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
from utils import load_dataset, build_preprocessor, encode_labels

def load_and_merge_datasets():
    """Load and merge all available datasets"""
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df3 = pd.DataFrame()
    
    if os.path.exists('ethiopian_hospital_dataset.xlsx'):
        df1 = pd.read_excel('ethiopian_hospital_dataset.xlsx')
        df1 = df1.rename(columns={column: column.replace('_', ' ') for column in df1.columns})
        df1 = df1.rename(columns={'Risk Level': 'Risk_Level', 'Length of Stay': 'Length_of_Stay'})
        print(f"✓ Loaded ethiopian_hospital_dataset.xlsx: {len(df1)} records")
        
    if os.path.exists('clinical_dataset.csv'):
        df2 = pd.read_csv('clinical_dataset.csv')
        rename_map = {
            'sex': 'Gender', 'age': 'Age', 'temperature': 'Temperature',
            'pulse': 'Heart Rate', 'target': 'Disease', 'weight': 'Weight',
            'height': 'Height', 'bmi': 'BMI', 'oxygen_saturation': 'Oxygen Saturation',
            'blood_pressure_systolic': 'Blood Pressure Systolic',
            'blood_pressure_diastolic': 'Blood Pressure Diastolic', 'pain_score': 'Pain Score'
        }
        df2 = df2.rename(columns=rename_map)
        print(f"✓ Loaded clinical_dataset.csv: {len(df2)} records")
        
    if os.path.exists('clinical_dataset.xlsx'):
        df3 = pd.read_excel('clinical_dataset.xlsx')
        df3 = df3.rename(columns=rename_map)
        print(f"✓ Loaded clinical_dataset.xlsx: {len(df3)} records")

    for df_new in [df2, df3]:
        if 'Gender' in df_new.columns:
            df_new['Gender'] = df_new['Gender'].str.capitalize()
            
    df = pd.concat([df1, df2, df3], ignore_index=True).drop_duplicates()
    df = df.loc[:, ~df.columns.duplicated()]
    print(f"✓ Combined dataset: {len(df)} total records\n")
    return df


def prepare_test_data(df):
    """Prepare test data matching training pipeline"""
    expected_features = [
        'Age', 'Gender', 'Region', 'Fever', 'Cough', 'Headache', 'Fatigue',
        'Vomiting', 'Diarrhea', 'Chest Pain', 'Shortness of Breath', 'Dizziness',
        'Temperature', 'Heart Rate', 'WBC Count', 'Hemoglobin', 'Malaria Test',
        'Comorbidity', 'Season', 'Weight', 'Height', 'BMI', 'Oxygen Saturation', 
        'Blood Pressure Systolic', 'Blood Pressure Diastolic', 'Pain Score'
    ]
    
    default_values = {
        'Chest Pain': 'No', 'Shortness of Breath': 'No', 'Dizziness': 'No',
        'WBC Count': np.nan, 'Hemoglobin': np.nan, 'Malaria Test': 'Unknown',
        'Weight': np.nan, 'Height': np.nan, 'BMI': np.nan,
        'Oxygen Saturation': np.nan, 'Blood Pressure Systolic': np.nan,
        'Blood Pressure Diastolic': np.nan, 'Pain Score': np.nan,
        'Disease': 'Unknown', 'Risk_Level': 'Low', 'Length_of_Stay': 0,
        'Gender': 'Unknown', 'Region': 'Unknown', 'Fever': 'Unknown',
        'Cough': 'Unknown', 'Headache': 'Unknown', 'Fatigue': 'Unknown',
        'Vomiting': 'Unknown', 'Diarrhea': 'Unknown', 'Comorbidity': 'Unknown', 'Season': 'Unknown'
    }
    
    for col, default in default_values.items():
        if col not in df.columns:
            df[col] = default

    df['Disease'] = df['Disease'].fillna('Unknown')
    df['Risk_Level'] = df['Risk_Level'].fillna('Low')
    
    # Group rare diseases
    disease_counts = df['Disease'].value_counts()
    rare_diseases = disease_counts[disease_counts < 50].index
    df.loc[df['Disease'].isin(rare_diseases), 'Disease'] = 'Other'

    X = df[expected_features].copy()
    y_disease, disease_encoder = encode_labels(df, 'Disease')
    y_risk, risk_encoder = encode_labels(df, 'Risk_Level')
    y_stay = df['Length_of_Stay'].replace({np.nan: 0}).astype(float)
    
    numeric_features = [
        'Age', 'Temperature', 'Heart Rate', 'WBC Count', 'Hemoglobin',
        'Weight', 'Height', 'BMI', 'Oxygen Saturation', 'Blood Pressure Systolic',
        'Blood Pressure Diastolic', 'Pain Score'
    ]
    categorical_features = [col for col in expected_features if col not in numeric_features]
    
    return X, y_disease, y_risk, y_stay, disease_encoder, risk_encoder, numeric_features, categorical_features


def evaluate_models():
    """Main evaluation function"""
    print("=" * 80)
    print("EthioHealth-AI: MODEL EVALUATION REPORT")
    print("=" * 80 + "\n")
    
    # Check if models exist
    required_models = [
        'models/disease_model.joblib',
        'models/risk_model.joblib',
        'models/stay_model.joblib',
        'models/preprocessor.joblib',
        'models/disease_label_encoder.joblib',
        'models/risk_label_encoder.joblib'
    ]
    
    missing = [m for m in required_models if not os.path.exists(m)]
    if missing:
        print("❌ ERROR: Missing required model files:")
        for m in missing:
            print(f"   - {m}")
        print("\nRun: python train.py")
        return
    
    print("📦 Loading Models & Data...\n")
    
    # Load datasets
    df = load_and_merge_datasets()
    X, y_disease, y_risk, y_stay, disease_enc, risk_enc, numeric_feat, categorical_feat = prepare_test_data(df)
    
    # Split data (same random state as training)
    X_train, X_test, y_d_train, y_d_test, y_r_train, y_r_test = train_test_split(
        X, y_disease, y_risk, test_size=0.2, random_state=42
    )
    X_train_stay, X_test_stay, y_s_train, y_s_test = train_test_split(
        X, y_stay, test_size=0.2, random_state=42
    )
    
    print(f"Test Set Size: {len(X_test)} records\n")
    
    # Load models
    disease_model = joblib.load('models/disease_model.joblib')
    risk_model = joblib.load('models/risk_model.joblib')
    stay_model = joblib.load('models/stay_model.joblib')
    
    # Make predictions
    print("🔮 Generating Predictions...\n")
    y_d_pred = disease_model.predict(X_test)
    y_r_pred = risk_model.predict(X_test)
    y_s_pred = stay_model.predict(X_test_stay)
    
    # ===== DISEASE PREDICTION RESULTS =====
    print("=" * 80)
    print("1️⃣  DISEASE PREDICTION MODEL")
    print("=" * 80)
    
    disease_labels = disease_enc.classes_
    report_disease = classification_report(y_d_test, y_d_pred, target_names=disease_labels, digits=4, zero_division=0)
    print("\nClassification Report:\n")
    print(report_disease)
    
    cm_disease = confusion_matrix(y_d_test, y_d_pred)
    print(f"Confusion Matrix Shape: {cm_disease.shape}")
    print(f"Total Correct Predictions: {np.trace(cm_disease)}")
    print(f"Total Incorrect Predictions: {len(y_d_test) - np.trace(cm_disease)}\n")
    
    # ===== RISK PREDICTION RESULTS =====
    print("\n" + "=" * 80)
    print("2️⃣  RISK LEVEL PREDICTION MODEL")
    print("=" * 80)
    
    risk_labels = risk_enc.classes_
    report_risk = classification_report(y_r_test, y_r_pred, target_names=risk_labels, digits=4, zero_division=0)
    print("\nClassification Report:\n")
    print(report_risk)
    
    cm_risk = confusion_matrix(y_r_test, y_r_pred)
    print(f"Confusion Matrix Shape: {cm_risk.shape}")
    print(f"Total Correct Predictions: {np.trace(cm_risk)}")
    print(f"Total Incorrect Predictions: {len(y_r_test) - np.trace(cm_risk)}\n")
    
    # ===== LENGTH OF STAY PREDICTION RESULTS =====
    print("\n" + "=" * 80)
    print("3️⃣  LENGTH OF STAY (LOS) PREDICTION MODEL")
    print("=" * 80)
    
    mse = mean_squared_error(y_s_test, y_s_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_s_test, y_s_pred)
    r2 = r2_score(y_s_test, y_s_pred)
    
    print(f"\nRegression Metrics:")
    print(f"  Mean Absolute Error (MAE):        {mae:.4f} days")
    print(f"  Root Mean Square Error (RMSE):   {rmse:.4f} days")
    print(f"  Mean Squared Error (MSE):        {mse:.4f}")
    print(f"  R² Score:                        {r2:.4f}")
    
    print(f"\nPrediction Range:")
    print(f"  Actual min/max:                  {y_s_test.min():.1f} - {y_s_test.max():.1f} days")
    print(f"  Predicted min/max:               {y_s_pred.min():.1f} - {y_s_pred.max():.1f} days")
    print(f"  Mean actual LOS:                 {y_s_test.mean():.1f} days")
    print(f"  Mean predicted LOS:              {y_s_pred.mean():.1f} days")
    
    # ===== SAMPLE PREDICTIONS =====
    print("\n" + "=" * 80)
    print("📋 SAMPLE PREDICTIONS (First 5 Test Cases)")
    print("=" * 80 + "\n")
    
    y_s_test_arr = np.array(y_s_test)
    for i in range(min(5, len(X_test))):
        print(f"Sample {i+1}:")
        print(f"  Disease:     {disease_labels[y_d_pred[i]]} (Actual: {disease_labels[y_d_test[i]]})")
        print(f"  Risk Level:  {risk_labels[y_r_pred[i]]} (Actual: {risk_labels[y_r_test[i]]})")
        print(f"  Stay (days): {y_s_pred[i]:.1f} (Actual: {y_s_test_arr[i]:.1f})")
        print()
    
    # ===== SUMMARY STATISTICS =====
    print("=" * 80)
    print("📊 OVERALL STATISTICS")
    print("=" * 80)
    
    from sklearn.metrics import accuracy_score
    disease_acc = accuracy_score(y_d_test, y_d_pred)
    risk_acc = accuracy_score(y_r_test, y_r_pred)
    
    print(f"\nModel Accuracy:")
    print(f"  Disease Prediction:    {disease_acc:.4f} ({disease_acc*100:.2f}%)")
    print(f"  Risk Level Prediction: {risk_acc:.4f} ({risk_acc*100:.2f}%)")
    print(f"  LOS Prediction (R²):   {r2:.4f}")
    
    print(f"\nTest Set Statistics:")
    print(f"  Total test samples:    {len(X_test)}")
    print(f"  Disease classes:       {len(disease_labels)}")
    print(f"  Risk classes:          {len(risk_labels)}")
    
    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    evaluate_models()
