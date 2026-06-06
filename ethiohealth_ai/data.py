import os
from pathlib import Path

import numpy as np
import pandas as pd

from ethiohealth_ai.config import DEFAULT_VALUES, EXPECTED_FEATURES


RENAME_MAP = {
    "sex": "Gender",
    "age": "Age",
    "temperature": "Temperature",
    "pulse": "Heart Rate",
    "target": "Disease",
    "weight": "Weight",
    "height": "Height",
    "bmi": "BMI",
    "oxygen_saturation": "Oxygen Saturation",
    "blood_pressure_systolic": "Blood Pressure Systolic",
    "blood_pressure_diastolic": "Blood Pressure Diastolic",
    "pain_score": "Pain Score",
    # Ethiopian dataset underscore variants
    "Chest_Pain": "Chest Pain",
    "Shortness_of_Breath": "Shortness of Breath",
    "Heart_Rate": "Heart Rate",
    "WBC_Count": "WBC Count",
    "Malaria_Test": "Malaria Test",
    "Hemoglobin": "Hemoglobin",
}


def load_dataset(data_path: str | os.PathLike) -> pd.DataFrame:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place your dataset in the project root "
            "or pass the correct --data path."
        )

    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    if df.empty:
        raise ValueError("The dataset is empty. Please provide a valid CSV or Excel file.")

    return normalize_columns(df)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={column: str(column).replace("_", " ") for column in df.columns})
    df = df.rename(columns={"Risk Level": "Risk_Level", "Length of Stay": "Length_of_Stay"})
    df = df.rename(columns=RENAME_MAP)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].astype(str).str.capitalize()
    return df


def load_and_merge_datasets(dataset_paths) -> pd.DataFrame:
    frames = []
    for path in dataset_paths:
        path = Path(path)
        if path.exists():
            frames.append(load_dataset(path))

    if not frames:
        raise ValueError("All datasets are missing or empty.")

    return pd.concat(frames, ignore_index=True).drop_duplicates()


def apply_default_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column, default in DEFAULT_VALUES.items():
        if column not in df.columns:
            df[column] = np.nan if default is None else default

    df["Disease"] = df["Disease"].fillna("Unknown")
    df["Risk_Level"] = df["Risk_Level"].fillna("Low")
    df["Length_of_Stay"] = df["Length_of_Stay"].replace({np.nan: 0}).astype(float)
    return df


def group_rare_diseases(df: pd.DataFrame, min_cases: int = 50) -> pd.DataFrame:
    df = df.copy()
    disease_counts = df["Disease"].value_counts()
    rare_diseases = disease_counts[disease_counts < min_cases].index
    df.loc[df["Disease"].isin(rare_diseases), "Disease"] = "Other"
    return df


def prepare_modeling_frame(df: pd.DataFrame):
    df = group_rare_diseases(apply_default_columns(df))
    x = df[EXPECTED_FEATURES].copy()
    y_stay = df["Length_of_Stay"].astype(float)
    return df, x, y_stay

