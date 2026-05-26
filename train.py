import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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


def train_models(data_path: str, output_dir: str):
    df = load_dataset(data_path)
    expected_features = [
        'Age', 'Gender', 'Region', 'Fever', 'Cough', 'Headache', 'Fatigue',
        'Vomiting', 'Diarrhea', 'Temperature', 'Heart Rate', 'Comorbidity', 'Season'
    ]
    for col in expected_features + ['Disease', 'Risk_Level', 'Length_of_Stay']:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: {col}")

    feature_columns = expected_features
    X = df[feature_columns].copy()
    y_disease, disease_encoder = encode_labels(df, 'Disease')
    y_risk, risk_encoder = encode_labels(df, 'Risk_Level')
    y_stay = df['Length_of_Stay'].replace({np.nan: 0}).astype(float)

    numeric_features = ['Age', 'Temperature', 'Heart Rate']
    categorical_features = [col for col in feature_columns if col not in numeric_features]

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    preprocessor.fit(X)
    X_processed = preprocessor.transform(X)
    feature_names = extract_feature_names(preprocessor, numeric_features, categorical_features)

    X_train_clf, X_test_clf, y_train_disease, y_test_disease = train_test_split(
        X_processed, y_disease, test_size=0.2, random_state=42, stratify=y_disease
    )
    _, _, y_train_risk, y_test_risk = train_test_split(
        X_processed, y_risk, test_size=0.2, random_state=42, stratify=y_risk
    )
    X_train_reg, X_test_reg, y_train_stay, y_test_stay = train_test_split(
        X_processed, y_stay, test_size=0.2, random_state=42
    )
    y_multi = np.column_stack((y_disease, y_risk))
    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
        X_processed, y_multi, test_size=0.2, random_state=42, stratify=y_disease
    )

    classification_models = {
        'logistic_regression': LogisticRegression(max_iter=500, random_state=42),
        'decision_tree': DecisionTreeClassifier(random_state=42),
        'svm': SVC(probability=True, random_state=42),
        'knn': KNeighborsClassifier(),
        'mlp': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
        'bagging': BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=8, random_state=42),
            n_estimators=50,
            max_samples=0.8,
            bootstrap=True,
            n_jobs=-1,
            random_state=42
        )
    }

    trained_clf_models = {}
    scores = {}
    print('\nTraining classification models for Disease Prediction...')
    for name, model in classification_models.items():
        model.fit(X_train_clf, y_train_disease)
        y_pred = model.predict(X_test_clf)
        report = evaluate_classification(y_test_disease, y_pred)
        scores[name] = report
        trained_clf_models[name] = model
        print(f"\n{name} results:")
        print(f"  accuracy: {report['accuracy']:.4f}")
        print(f"  precision: {report['precision']:.4f}")
        print(f"  recall: {report['recall']:.4f}")
        print(f"  f1_score: {report['f1_score']:.4f}")

    best_clf_name = max(scores, key=lambda k: scores[k]['f1_score'])
    best_disease_model = trained_clf_models[best_clf_name]
    print(f"\nSelected best disease model: {best_clf_name}")

    print('\nTraining classification models for Risk Level Prediction...')
    trained_risk_models = {}
    risk_scores = {}
    for name, model in classification_models.items():
        model.fit(X_train_clf, y_train_risk)
        y_pred_risk = model.predict(X_test_clf)
        report = evaluate_classification(y_test_risk, y_pred_risk)
        risk_scores[name] = report
        trained_risk_models[name] = model
        print(f"\n{name} risk results:")
        print(f"  accuracy: {report['accuracy']:.4f}")
        print(f"  precision: {report['precision']:.4f}")
        print(f"  recall: {report['recall']:.4f}")
        print(f"  f1_score: {report['f1_score']:.4f}")

    best_risk_name = max(risk_scores, key=lambda k: risk_scores[k]['f1_score'])
    best_risk_model = trained_risk_models[best_risk_name]
    print(f"\nSelected best risk model: {best_risk_name}")

    print('\nTraining multi-task model for Disease and Risk Level Prediction...')
    multi_task_model = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42),
        n_jobs=-1
    )
    multi_task_model.fit(X_train_multi, y_train_multi)
    y_pred_multi = multi_task_model.predict(X_test_multi)
    multi_disease_report = evaluate_classification(y_test_multi[:, 0], y_pred_multi[:, 0])
    multi_risk_report = evaluate_classification(y_test_multi[:, 1], y_pred_multi[:, 1])
    print('  disease task:')
    print(f"    accuracy: {multi_disease_report['accuracy']:.4f}")
    print(f"    precision: {multi_disease_report['precision']:.4f}")
    print(f"    recall: {multi_disease_report['recall']:.4f}")
    print(f"    f1_score: {multi_disease_report['f1_score']:.4f}")
    print('  risk task:')
    print(f"    accuracy: {multi_risk_report['accuracy']:.4f}")
    print(f"    precision: {multi_risk_report['precision']:.4f}")
    print(f"    recall: {multi_risk_report['recall']:.4f}")
    print(f"    f1_score: {multi_risk_report['f1_score']:.4f}")

    print('\nTraining regression model for Length of Stay Prediction...')
    stay_model = RandomForestRegressor(n_estimators=150, random_state=42)
    stay_model.fit(X_train_reg, y_train_stay)
    y_pred_stay = stay_model.predict(X_test_reg)
    stay_report = evaluate_regression(y_test_stay, y_pred_stay)
    print(f"  mse: {stay_report['mse']:.4f}")
    print(f"  rmse: {stay_report['rmse']:.4f}")
    print(f"  mae: {stay_report['mae']:.4f}")
    print(f"  r2: {stay_report['r2']:.4f}")

    print('\nTraining K-Means clustering model...')
    kmeans_model = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans_model.fit(X_processed)
    kmeans_labels = kmeans_model.labels_
    print(f"  KMeans cluster sizes: {np.bincount(kmeans_labels)}")

    print('\nTraining optional DBSCAN clustering model...')
    dbscan_model = DBSCAN(eps=0.75, min_samples=8)
    dbscan_model.fit(X_processed)
    dbscan_labels = dbscan_model.labels_
    unique_labels = np.unique(dbscan_labels)
    print(f"  DBSCAN unique clusters: {unique_labels}")

    model_dir = os.path.join(output_dir, 'models')
    save_model(preprocessor, os.path.join(model_dir, 'preprocessor.joblib'))
    save_model(disease_encoder, os.path.join(model_dir, 'disease_label_encoder.joblib'))
    save_model(risk_encoder, os.path.join(model_dir, 'risk_label_encoder.joblib'))
    save_model(best_disease_model, os.path.join(model_dir, 'disease_model.joblib'))
    save_model(best_risk_model, os.path.join(model_dir, 'risk_model.joblib'))
    save_model(multi_task_model, os.path.join(model_dir, 'multi_task_model.joblib'))
    save_model(stay_model, os.path.join(model_dir, 'stay_model.joblib'))
    save_model(kmeans_model, os.path.join(model_dir, 'kmeans_model.joblib'))
    save_model(dbscan_model, os.path.join(model_dir, 'dbscan_model.joblib'))
    save_model(feature_names, os.path.join(model_dir, 'feature_names.joblib'))

    print('\nTraining complete. Models and preprocessing artifacts are saved in the models folder.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train EthioHealth-AI models from CSV data.')
    parser.add_argument('--data', type=str, default='ethiopia_health_data.csv',
                        help='Path to the training CSV file.')
    parser.add_argument('--output', type=str, default='.', help='Root output folder for saved models.')
    args = parser.parse_args()
    train_models(args.data, args.output)
