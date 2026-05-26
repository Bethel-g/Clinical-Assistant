import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans, DBSCAN

from utils import (
    load_dataset,
    build_preprocessor,
    extract_feature_names,
    encode_labels,
    evaluate_classification,
    evaluate_regression
)


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Saved model: {path}")


def most_common_by_disease(df, value_column):
    if value_column not in df.columns:
        return {}
    mapping = {}
    for disease, group in df.groupby('Disease'):
        values = group[value_column].dropna().astype(str).str.strip()
        values = values[values != '']
        if not values.empty:
            mapping[str(disease)] = values.mode().iloc[0]
    return mapping


def train_models(output_dir: str):
    df = pd.DataFrame()
    if os.path.exists('ethiopian_hospital_dataset.xlsx'):
        df = pd.read_excel('ethiopian_hospital_dataset.xlsx')
    
    if df.empty:
        raise ValueError("ethiopian_hospital_dataset.xlsx is missing or empty.")
    
    df = df.rename(columns={column: column.replace('_', ' ') for column in df.columns})
    df = df.rename(columns={
        'Risk Level': 'Risk_Level',
        'Length of Stay': 'Length_of_Stay'
    })

    expected_features = [
        'Age', 'Gender', 'Region', 'Fever', 'Cough', 'Headache', 'Fatigue',
        'Vomiting', 'Diarrhea', 'Chest Pain', 'Shortness of Breath', 'Dizziness',
        'Temperature', 'Heart Rate', 'WBC Count', 'Hemoglobin', 'Malaria Test',
        'Comorbidity', 'Season'
    ]
    default_values = {
        'Chest Pain': 'No',
        'Shortness of Breath': 'No',
        'Dizziness': 'No',
        'WBC Count': np.nan,
        'Hemoglobin': np.nan,
        'Malaria Test': 'Unknown'
    }
    for col, default in default_values.items():
        if col not in df.columns:
            df[col] = default
    for col in expected_features + ['Disease', 'Risk_Level', 'Length_of_Stay']:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: {col}")

    X = df[expected_features].copy()
    y_disease, disease_encoder = encode_labels(df, 'Disease')
    y_risk, risk_encoder = encode_labels(df, 'Risk_Level')
    y_stay = df['Length_of_Stay'].replace({np.nan: 0}).astype(float)

    numeric_features = ['Age', 'Temperature', 'Heart Rate', 'WBC Count', 'Hemoglobin']
    categorical_features = [col for col in expected_features if col not in numeric_features]

    # Preprocessor
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    
    # We must fit preprocessor to extract feature names
    preprocessor.fit(X)
    feature_names = extract_feature_names(preprocessor, numeric_features, categorical_features)

    # 2. Split training and testing with 80 20
    X_train, X_test, y_train_disease, y_test_disease, y_train_risk, y_test_risk = train_test_split(
        X, y_disease, y_risk, test_size=0.2, random_state=42, stratify=y_disease
    )
    
    X_train_stay, X_test_stay, y_train_stay, y_test_stay = train_test_split(
        X, y_stay, test_size=0.2, random_state=42
    )
    
    y_multi = np.column_stack((y_disease, y_risk))
    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
        X, y_multi, test_size=0.2, random_state=42, stratify=y_disease
    )

    # 3. Use proper ML pipeline model
    classification_models = {
        'logistic_regression': LogisticRegression(max_iter=500, random_state=42),
        'decision_tree': DecisionTreeClassifier(random_state=42),
        'mlp': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
        'bagging': BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=8, random_state=42),
            n_estimators=50, max_samples=0.8, bootstrap=True, n_jobs=-1, random_state=42
        )
    }

    # Disease Models
    trained_clf_models = {}
    scores = {}
    print('\nTraining classification pipeline models for Disease Prediction...')
    for name, clf in classification_models.items():
        pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', clf)])
        pipeline.fit(X_train, y_train_disease)
        y_pred = pipeline.predict(X_test)
        report = evaluate_classification(y_test_disease, y_pred)
        scores[name] = report
        trained_clf_models[name] = pipeline
        print(f"\n{name} results:")
        print(f"  accuracy: {report['accuracy']:.4f}")
        print(f"  f1_score: {report['f1_score']:.4f}")

    best_clf_name = max(scores, key=lambda k: scores[k]['f1_score'])
    best_disease_model = trained_clf_models[best_clf_name]
    print(f"\nSelected best disease pipeline: {best_clf_name}")

    # Risk Models
    trained_risk_models = {}
    risk_scores = {}
    print('\nTraining classification pipeline models for Risk Level Prediction...')
    for name, clf in classification_models.items():
        pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', clf)])
        pipeline.fit(X_train, y_train_risk)
        y_pred_risk = pipeline.predict(X_test)
        report = evaluate_classification(y_test_risk, y_pred_risk)
        risk_scores[name] = report
        trained_risk_models[name] = pipeline
        print(f"\n{name} risk results:")
        print(f"  accuracy: {report['accuracy']:.4f}")
        print(f"  f1_score: {report['f1_score']:.4f}")

    best_risk_name = max(risk_scores, key=lambda k: risk_scores[k]['f1_score'])
    best_risk_model = trained_risk_models[best_risk_name]
    print(f"\nSelected best risk pipeline: {best_risk_name}")

    # Multi-task
    print('\nTraining multi-task pipeline...')
    multi_task_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', MultiOutputClassifier(RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42), n_jobs=-1))
    ])
    multi_task_pipeline.fit(X_train_multi, y_train_multi)
    y_pred_multi = multi_task_pipeline.predict(X_test_multi)
    multi_disease_report = evaluate_classification(y_test_multi[:, 0], y_pred_multi[:, 0])
    multi_risk_report = evaluate_classification(y_test_multi[:, 1], y_pred_multi[:, 1])

    # Stay (Regression)
    print('\nTraining regression pipeline...')
    stay_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=20, max_depth=6, n_jobs=-1, random_state=42))
    ])
    stay_pipeline.fit(X_train_stay, y_train_stay)
    y_pred_stay = stay_pipeline.predict(X_test_stay)
    stay_report = evaluate_regression(y_test_stay, y_pred_stay)
    print(f"  rmse: {stay_report['rmse']:.4f}")

    recommendation_maps = {
        'treatment_by_disease': most_common_by_disease(df, 'Treatment'),
        'lab_order_by_disease': most_common_by_disease(df, 'Lab Order')
    }
    training_metrics = {
        'best_disease_model': best_clf_name,
        'best_disease_f1': float(scores[best_clf_name]['f1_score']),
        'best_disease_accuracy': float(scores[best_clf_name]['accuracy']),
        'best_risk_model': best_risk_name,
        'best_risk_f1': float(risk_scores[best_risk_name]['f1_score']),
        'best_risk_accuracy': float(risk_scores[best_risk_name]['accuracy']),
        'multi_task_disease_f1': float(multi_disease_report['f1_score']),
        'multi_task_risk_f1': float(multi_risk_report['f1_score']),
        'stay_rmse': float(stay_report['rmse']),
        'stay_mae': float(stay_report['mae']),
        'training_rows': int(len(df)),
    }

    print('\nTraining clustering models...')
    # Clustering cannot easily be in a standard supervised Pipeline because KMeans doesn't use y. 
    # But we can just use the preprocessor.
    X_processed = preprocessor.transform(X)
    kmeans_model = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans_model.fit(X_processed)
    
    dbscan_model = DBSCAN(eps=0.75, min_samples=8)
    dbscan_model.fit(X_processed)

    model_dir = os.path.join(output_dir, 'models')
    save_model(preprocessor, os.path.join(model_dir, 'preprocessor.joblib'))
    save_model(disease_encoder, os.path.join(model_dir, 'disease_label_encoder.joblib'))
    save_model(risk_encoder, os.path.join(model_dir, 'risk_label_encoder.joblib'))
    save_model(best_disease_model, os.path.join(model_dir, 'disease_model.joblib'))
    save_model(best_risk_model, os.path.join(model_dir, 'risk_model.joblib'))
    save_model(multi_task_pipeline, os.path.join(model_dir, 'multi_task_model.joblib'))
    save_model(stay_pipeline, os.path.join(model_dir, 'stay_model.joblib'))
    save_model(kmeans_model, os.path.join(model_dir, 'kmeans_model.joblib'))
    save_model(dbscan_model, os.path.join(model_dir, 'dbscan_model.joblib'))
    save_model(feature_names, os.path.join(model_dir, 'feature_names.joblib'))
    save_model(recommendation_maps, os.path.join(model_dir, 'recommendation_maps.joblib'))
    save_model(training_metrics, os.path.join(model_dir, 'training_metrics.joblib'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='.')
    args = parser.parse_args()
    train_models(args.output)
