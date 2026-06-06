import argparse
import os

import joblib
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from ethiohealth_ai.config import CATEGORICAL_FEATURES, DATASET_FILES, NUMERIC_FEATURES
from ethiohealth_ai.data import load_and_merge_datasets, prepare_modeling_frame
from ethiohealth_ai.ml import (
    build_preprocessor,
    encode_labels,
    evaluate_classification,
    evaluate_regression,
    extract_feature_names,
)


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Saved model: {path}")


def most_common_by_disease(df, value_column):
    if value_column not in df.columns:
        return {}
    mapping = {}
    for disease, group in df.groupby("Disease"):
        values = group[value_column].dropna().astype(str).str.strip()
        values = values[values != ""]
        if not values.empty:
            mapping[str(disease)] = values.mode().iloc[0]
    return mapping


def train_models(output_dir: str):
    df = load_and_merge_datasets(DATASET_FILES)
    df, x, _ = prepare_modeling_frame(df)
    y_disease, disease_encoder = encode_labels(df, "Disease")
    y_risk, risk_encoder = encode_labels(df, "Risk_Level")

    x_train, x_test, y_train_disease, y_test_disease, y_train_risk, y_test_risk = train_test_split(
        x, y_disease, y_risk, test_size=0.2, random_state=42
    )
    y_multi = np.column_stack((y_disease, y_risk))
    x_train_multi, x_test_multi, y_train_multi, y_test_multi = train_test_split(
        x, y_multi, test_size=0.2, random_state=42
    )

    classification_models = {
        "decision_tree": DecisionTreeClassifier(max_depth=15, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=30, max_depth=10, n_jobs=-1, random_state=42
        ),
    }

    trained_clf_models = {}
    scores = {}
    print("\nTraining classification pipeline models for Disease Prediction...")
    for name, clf in classification_models.items():
        pipeline = Pipeline(
            [("preprocessor", build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)), ("classifier", clf)]
        )
        pipeline.fit(x_train, y_train_disease)
        report = evaluate_classification(y_test_disease, pipeline.predict(x_test))
        scores[name] = report
        trained_clf_models[name] = pipeline
        print(f"\n{name} results:")
        print(f"  accuracy: {report['accuracy']:.4f}")
        print(f"  f1_score: {report['f1_score']:.4f}")

    best_clf_name = max(scores, key=lambda key: scores[key]["f1_score"])
    best_disease_model = trained_clf_models[best_clf_name]
    print(f"\nSelected best disease pipeline: {best_clf_name}")

    trained_risk_models = {}
    risk_scores = {}
    print("\nTraining classification pipeline models for Risk Level Prediction...")
    for name, clf in classification_models.items():
        pipeline = Pipeline(
            [("preprocessor", build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)), ("classifier", clf)]
        )
        pipeline.fit(x_train, y_train_risk)
        report = evaluate_classification(y_test_risk, pipeline.predict(x_test))
        risk_scores[name] = report
        trained_risk_models[name] = pipeline
        print(f"\n{name} risk results:")
        print(f"  accuracy: {report['accuracy']:.4f}")
        print(f"  f1_score: {report['f1_score']:.4f}")

    best_risk_name = max(risk_scores, key=lambda key: risk_scores[key]["f1_score"])
    best_risk_model = trained_risk_models[best_risk_name]
    print(f"\nSelected best risk pipeline: {best_risk_name}")

    print("\nTraining multi-task pipeline...")
    multi_task_pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)),
            (
                "classifier",
                MultiOutputClassifier(
                    RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42),
                    n_jobs=-1,
                ),
            ),
        ]
    )
    multi_task_pipeline.fit(x_train_multi, y_train_multi)
    y_pred_multi = multi_task_pipeline.predict(x_test_multi)
    multi_disease_report = evaluate_classification(y_test_multi[:, 0], y_pred_multi[:, 0])
    multi_risk_report = evaluate_classification(y_test_multi[:, 1], y_pred_multi[:, 1])

    recommendation_maps = {
        "treatment_by_disease": most_common_by_disease(df, "Treatment"),
        "lab_order_by_disease": most_common_by_disease(df, "Lab Order"),
    }
    training_metrics = {
        "best_disease_model": best_clf_name,
        "best_disease_f1": float(scores[best_clf_name]["f1_score"]),
        "best_disease_accuracy": float(scores[best_clf_name]["accuracy"]),
        "best_risk_model": best_risk_name,
        "best_risk_f1": float(risk_scores[best_risk_name]["f1_score"]),
        "best_risk_accuracy": float(risk_scores[best_risk_name]["accuracy"]),
        "multi_task_disease_f1": float(multi_disease_report["f1_score"]),
        "multi_task_risk_f1": float(multi_risk_report["f1_score"]),
        "training_rows": int(len(df)),
    }

    print("\nTraining clustering models...")
    clustering_preprocessor = build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    x_processed = clustering_preprocessor.fit_transform(x)
    kmeans_model = KMeans(n_clusters=4, random_state=42, n_init=10).fit(x_processed)
    dbscan_model = DBSCAN(eps=0.75, min_samples=8).fit(x_processed)

    feature_names = extract_feature_names(
        best_disease_model.named_steps["preprocessor"], NUMERIC_FEATURES, CATEGORICAL_FEATURES
    )
    model_dir = os.path.join(output_dir, "models")
    save_model(best_disease_model.named_steps["preprocessor"], os.path.join(model_dir, "preprocessor.joblib"))
    save_model(disease_encoder, os.path.join(model_dir, "disease_label_encoder.joblib"))
    save_model(risk_encoder, os.path.join(model_dir, "risk_label_encoder.joblib"))
    save_model(best_disease_model, os.path.join(model_dir, "disease_model.joblib"))
    save_model(best_risk_model, os.path.join(model_dir, "risk_model.joblib"))
    save_model(multi_task_pipeline, os.path.join(model_dir, "multi_task_model.joblib"))
    save_model(kmeans_model, os.path.join(model_dir, "kmeans_model.joblib"))
    save_model(dbscan_model, os.path.join(model_dir, "dbscan_model.joblib"))
    save_model(feature_names, os.path.join(model_dir, "feature_names.joblib"))
    save_model(recommendation_maps, os.path.join(model_dir, "recommendation_maps.joblib"))
    save_model(training_metrics, os.path.join(model_dir, "training_metrics.joblib"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=".")
    args = parser.parse_args()
    train_models(args.output)

