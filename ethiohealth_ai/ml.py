import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def extract_feature_names(preprocessor, numeric_features, categorical_features):
    cat_transformer = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = cat_transformer.get_feature_names_out(categorical_features).tolist()
    return list(numeric_features) + cat_names


def encode_labels(df, column: str):
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(df[column].astype(str).fillna("Unknown"))
    return encoded, encoder


def evaluate_classification(y_true, y_pred, labels=None):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }


def evaluate_regression(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": mse,
        "rmse": np.sqrt(mse),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def explain_prediction(model, input_array, feature_names, top_n=5):
    if hasattr(model, "named_steps"):
        model = model.steps[-1][1]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "estimators_") and model.estimators_:
        tree_importances = [
            estimator.feature_importances_
            for estimator in model.estimators_
            if hasattr(estimator, "feature_importances_")
        ]
        if not tree_importances:
            return ["Model does not expose a direct explanation method."]
        importances = np.mean(tree_importances, axis=0)
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            coef = np.mean(np.abs(coef), axis=0)
        importances = np.abs(coef)
    else:
        return ["Model does not expose a direct explanation method."]

    if len(importances) != len(feature_names):
        return ["Feature importance could not be aligned with feature names."]

    indices = np.argsort(importances)[::-1][:top_n]
    return [f"{feature_names[idx]}: importance {importances[idx]:.3f}" for idx in indices]


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

