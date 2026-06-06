import joblib
import numpy as np
import pandas as pd

from ethiohealth_ai.config import MODEL_DIR, OPTIONAL_ARTIFACTS, REQUIRED_ARTIFACTS


def load_artifacts(model_dir=MODEL_DIR):
    artifacts = {}
    for filename in REQUIRED_ARTIFACTS:
        path = model_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Required model artifact not found: {path}")
        artifacts[filename] = joblib.load(path)

    for filename in OPTIONAL_ARTIFACTS:
        path = model_dir / filename
        artifacts[filename] = joblib.load(path) if path.exists() else {}

    return artifacts


def build_input_dataframe(inputs):
    return pd.DataFrame([inputs])


def get_confidence(prediction_proba):
    if prediction_proba is None:
        return 0.0
    return float(np.max(prediction_proba))


def get_recommendation(recommendation_maps, key, disease_label):
    value = recommendation_maps.get(key, {}).get(disease_label)
    return value if value else "Review local clinical protocol"


def build_ai_reason(inputs, disease_label):
    reasons = []
    if inputs.get("Temperature", 0) >= 38:
        reasons.append("high temperature")
    if inputs.get("Cough") == "Yes":
        reasons.append("cough")
    if inputs.get("Chest Pain") == "Yes":
        reasons.append("chest pain")
    if inputs.get("Shortness of Breath") == "Yes":
        reasons.append("shortness of breath")
    if inputs.get("WBC Count", 0) >= 11000:
        reasons.append("high WBC count")
    if inputs.get("Malaria Test") == "Positive":
        reasons.append("positive malaria test")
    if not reasons:
        reasons.append("the submitted symptoms, vitals, and lab results")
    return f"{' + '.join(reasons)} -> {disease_label}"

