from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

DATASET_FILES = (
    PROJECT_ROOT / "ethiopian_hospital_dataset.xlsx",
)

EXPECTED_FEATURES = [
    "Age",
    "Gender",
    "Fever",
    "Cough",
    "Headache",
    "Fatigue",
    "Vomiting",
    "Diarrhea",
    "Chest Pain",
    "Shortness of Breath",
    "Dizziness",
    "Temperature",
    "Heart Rate",
    "WBC Count",
    "Hemoglobin",
    "Malaria Test",
    "Comorbidity",
    "Season",
]

NUMERIC_FEATURES = [
    "Age",
    "Temperature",
    "Heart Rate",
    "WBC Count",
    "Hemoglobin",
]

CATEGORICAL_FEATURES = [
    feature for feature in EXPECTED_FEATURES if feature not in NUMERIC_FEATURES
]

DEFAULT_VALUES = {
    "Chest Pain": "No",
    "Shortness of Breath": "No",
    "Dizziness": "No",
    "WBC Count": None,
    "Hemoglobin": None,
    "Malaria Test": "Unknown",
    "Weight": None,
    "Height": None,
    "BMI": None,
    "Oxygen Saturation": None,
    "Blood Pressure Systolic": None,
    "Blood Pressure Diastolic": None,
    "Pain Score": None,
    "Disease": "Unknown",
    "Risk_Level": "Low",
    "Length_of_Stay": 0,
    "Gender": "Unknown",
    "Region": "Unknown",
    "Fever": "Unknown",
    "Cough": "Unknown",
    "Headache": "Unknown",
    "Fatigue": "Unknown",
    "Vomiting": "Unknown",
    "Diarrhea": "Unknown",
    "Comorbidity": "Unknown",
    "Season": "Unknown",
}

REQUIRED_ARTIFACTS = [
    "preprocessor.joblib",
    "disease_model.joblib",
    "risk_model.joblib",
    "disease_label_encoder.joblib",
    "risk_label_encoder.joblib",
    "feature_names.joblib",
]

OPTIONAL_ARTIFACTS = [
    "recommendation_maps.joblib",
    "training_metrics.joblib",
]

