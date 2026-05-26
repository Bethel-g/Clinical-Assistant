import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score


def load_dataset(data_path: str) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Place your dataset in the project root or pass the correct --data path."
        )
    _, extension = os.path.splitext(data_path.lower())
    if extension in {'.xlsx', '.xls'}:
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)
    df = df.rename(columns={column: column.replace('_', ' ') for column in df.columns})
    df = df.rename(columns={
        'Risk Level': 'Risk_Level',
        'Length of Stay': 'Length_of_Stay'
    })
    if df.empty:
        raise ValueError("The dataset is empty. Please provide a valid CSV or Excel file.")
    return df


def build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    return preprocessor


def extract_feature_names(preprocessor, numeric_features, categorical_features):
    numeric_names = numeric_features
    cat_transformer = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_names = cat_transformer.get_feature_names_out(categorical_features).tolist()
    return numeric_names + cat_names


def encode_labels(df: pd.DataFrame, column: str):
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(df[column].astype(str).fillna('Unknown'))
    return encoded, encoder


def evaluate_classification(y_true, y_pred, labels=None):
    report = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels)
    }
    return report


def evaluate_regression(y_true, y_pred):
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }


def explain_prediction(model, input_array, feature_names, top_n=5):
    if hasattr(model, 'named_steps'):
        # Extract the final estimator from the pipeline
        model = model.steps[-1][1]

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'estimators_') and model.estimators_:
        tree_importances = [
            estimator.feature_importances_
            for estimator in model.estimators_
            if hasattr(estimator, 'feature_importances_')
        ]
        if not tree_importances:
            return ["Model does not expose a direct explanation method."]
        importances = np.mean(tree_importances, axis=0)
    elif hasattr(model, 'coef_'):
        coef = model.coef_
        if coef.ndim > 1:
            coef = np.mean(np.abs(coef), axis=0)
        importances = np.abs(coef)
    else:
        return ["Model does not expose a direct explanation method."]

    if len(importances) != len(feature_names):
        return ["Feature importance could not be aligned with feature names."]

    indices = np.argsort(importances)[::-1][:top_n]
    explanation = []
    for idx in indices:
        explanation.append(f"{feature_names[idx]}: importance {importances[idx]:.3f}")
    return explanation


def validate_numeric_value(value, name, min_value=None, max_value=None):
    if value is None:
        raise ValueError(f"{name} is required.")
    try:
        value = float(value)
    except ValueError:
        raise ValueError(f"{name} must be a numeric value.")
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be at least {min_value}.")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must be at most {max_value}.")
    return value
