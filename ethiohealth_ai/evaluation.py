import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from ethiohealth_ai.config import DATASET_FILES, MODEL_DIR
from ethiohealth_ai.data import load_and_merge_datasets, prepare_modeling_frame
from ethiohealth_ai.ml import encode_labels


def evaluate_models():
    print("=" * 80)
    print("EthioHealth-AI: MODEL EVALUATION REPORT")
    print("=" * 80 + "\n")

    required_models = [
        MODEL_DIR / "disease_model.joblib",
        MODEL_DIR / "risk_model.joblib",
        MODEL_DIR / "preprocessor.joblib",
        MODEL_DIR / "disease_label_encoder.joblib",
        MODEL_DIR / "risk_label_encoder.joblib",
    ]
    missing = [str(path) for path in required_models if not path.exists()]
    if missing:
        print("ERROR: Missing required model files:")
        for path in missing:
            print(f"   - {path}")
        print("\nRun: python train.py")
        return

    print("Loading Models & Data...\n")
    df = load_and_merge_datasets(DATASET_FILES)
    df, x, _ = prepare_modeling_frame(df)
    y_disease, disease_encoder = encode_labels(df, "Disease")
    y_risk, risk_encoder = encode_labels(df, "Risk_Level")

    _, x_test, _, y_d_test, _, y_r_test = train_test_split(
        x, y_disease, y_risk, test_size=0.2, random_state=42
    )

    print(f"Test Set Size: {len(x_test)} records\n")

    disease_model = joblib.load(MODEL_DIR / "disease_model.joblib")
    risk_model = joblib.load(MODEL_DIR / "risk_model.joblib")

    print("Generating Predictions...\n")
    y_d_pred = disease_model.predict(x_test)
    y_r_pred = risk_model.predict(x_test)

    print("=" * 80)
    print("1. DISEASE PREDICTION MODEL")
    print("=" * 80)
    disease_labels = disease_encoder.classes_
    print("\nClassification Report:\n")
    print(classification_report(y_d_test, y_d_pred, target_names=disease_labels, digits=4, zero_division=0))
    cm_disease = confusion_matrix(y_d_test, y_d_pred)
    print(f"Confusion Matrix Shape: {cm_disease.shape}")
    print(f"Total Correct Predictions: {np.trace(cm_disease)}")
    print(f"Total Incorrect Predictions: {len(y_d_test) - np.trace(cm_disease)}\n")

    print("\n" + "=" * 80)
    print("2. RISK LEVEL PREDICTION MODEL")
    print("=" * 80)
    risk_labels = risk_encoder.classes_
    print("\nClassification Report:\n")
    print(classification_report(y_r_test, y_r_pred, target_names=risk_labels, digits=4, zero_division=0))
    cm_risk = confusion_matrix(y_r_test, y_r_pred)
    print(f"Confusion Matrix Shape: {cm_risk.shape}")
    print(f"Total Correct Predictions: {np.trace(cm_risk)}")
    print(f"Total Incorrect Predictions: {len(y_r_test) - np.trace(cm_risk)}\n")

    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS (First 5 Test Cases)")
    print("=" * 80 + "\n")
    for index in range(min(5, len(x_test))):
        print(f"Sample {index + 1}:")
        print(f"  Disease:     {disease_labels[y_d_pred[index]]} (Actual: {disease_labels[y_d_test[index]]})")
        print(f"  Risk Level:  {risk_labels[y_r_pred[index]]} (Actual: {risk_labels[y_r_test[index]]})")
        print()

    disease_acc = accuracy_score(y_d_test, y_d_pred)
    risk_acc = accuracy_score(y_r_test, y_r_pred)
    print("=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    print("\nModel Accuracy:")
    print(f"  Disease Prediction:    {disease_acc:.4f} ({disease_acc * 100:.2f}%)")
    print(f"  Risk Level Prediction: {risk_acc:.4f} ({risk_acc * 100:.2f}%)")
    print(f"\nTest Set Statistics:")
    print(f"  Total test samples:    {len(x_test)}")
    print(f"  Disease classes:       {len(disease_labels)}")
    print(f"  Risk classes:          {len(risk_labels)}")
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80 + "\n")


def main():
    evaluate_models()

