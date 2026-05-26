import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


SYMPTOM_COLUMNS = ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptoms']
REQUIRED_COLUMNS = [
    'Age', 'Gender', 'Region', 'Fever', 'Cough', 'Headache', 'Fatigue',
    'Vomiting', 'Diarrhea', 'Temperature', 'Heart Rate', 'Comorbidity',
    'Season', 'Disease', 'Risk_Level', 'Length_of_Stay'
]


def find_column(columns, candidates):
    normalized = {column.lower().strip().replace(' ', '_'): column for column in columns}
    for candidate in candidates:
        key = candidate.lower().strip().replace(' ', '_')
        if key in normalized:
            return normalized[key]
    return None


def has_symptom(row, symptom):
    values = []
    for column in SYMPTOM_COLUMNS:
        if column in row and pd.notna(row[column]):
            values.append(str(row[column]).lower())
    return 'Yes' if symptom.lower() in ' '.join(values) else 'No'


def normalize_risk(value):
    if pd.isna(value):
        return 'Medium'
    value = str(value).strip().lower()
    if value in {'mild', 'low', 'normal'}:
        return 'Low'
    if value in {'moderate', 'medium'}:
        return 'Medium'
    if value in {'severe', 'high', 'critical'}:
        return 'High'
    return 'Medium'


def estimate_length_of_stay(row):
    risk = row['Risk_Level']
    treatment = str(row.get('Treatment Plan', '')).lower()
    temperature = row.get('Temperature', 37.0)
    heart_rate = row.get('Heart Rate', 80)

    if risk == 'High' or 'hospital' in treatment:
        base_days = 7
    elif risk == 'Medium':
        base_days = 4
    else:
        base_days = 2

    if temperature >= 39:
        base_days += 1
    if heart_rate >= 110:
        base_days += 1
    return int(np.clip(base_days, 1, 14))


def prepare_dataset(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError(f'Input CSV is empty: {input_csv}')

    age_col = find_column(df.columns, ['Age'])
    gender_col = find_column(df.columns, ['Gender', 'Sex'])
    temp_col = find_column(df.columns, ['Body Temperature', 'Temperature', 'Body_Temperature'])
    heart_rate_col = find_column(df.columns, ['Heart Rate', 'Heart_Rate'])
    diagnosis_col = find_column(df.columns, ['Diagnosis', 'Disease'])
    severity_col = find_column(df.columns, ['Severity', 'Risk_Level', 'Risk Level'])
    treatment_col = find_column(df.columns, ['Treatment Plan', 'Treatment_Plan', 'Treatment'])

    missing = [
        name for name, column in {
            'Age': age_col,
            'Gender': gender_col,
            'Temperature': temp_col,
            'Heart Rate': heart_rate_col,
            'Diagnosis/Disease': diagnosis_col,
            'Severity/Risk': severity_col
        }.items()
        if column is None
    ]
    if missing:
        raise ValueError(f'Missing required Kaggle columns: {", ".join(missing)}')

    prepared = pd.DataFrame()
    prepared['Age'] = pd.to_numeric(df[age_col], errors='coerce').fillna(df[age_col].median())
    prepared['Gender'] = df[gender_col].astype(str).str.title().replace({'Nan': 'Other'})
    prepared['Region'] = 'Unknown'

    for symptom in ['Fever', 'Cough', 'Headache', 'Fatigue', 'Vomiting', 'Diarrhea']:
        prepared[symptom] = df.apply(lambda row: has_symptom(row, symptom), axis=1)

    prepared['Temperature'] = pd.to_numeric(df[temp_col], errors='coerce').fillna(37.0)
    prepared['Heart Rate'] = pd.to_numeric(df[heart_rate_col], errors='coerce').fillna(80).astype(int)
    prepared['Comorbidity'] = 'No'
    prepared['Season'] = 'Unknown'
    prepared['Disease'] = df[diagnosis_col].astype(str).str.strip()
    prepared['Risk_Level'] = df[severity_col].apply(normalize_risk)

    if treatment_col:
        prepared['Treatment Plan'] = df[treatment_col]
    else:
        prepared['Treatment Plan'] = ''
    prepared['Length_of_Stay'] = prepared.apply(estimate_length_of_stay, axis=1)
    prepared = prepared[REQUIRED_COLUMNS]

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)
    print(f'Saved prepared dataset: {output_path}')
    print(f'Rows: {len(prepared)}')
    print(f'Columns: {", ".join(prepared.columns)}')


def main():
    parser = argparse.ArgumentParser(
        description='Convert the Kaggle Disease Diagnosis Dataset into the EthioHealth-AI training schema.'
    )
    parser.add_argument('--input', required=True, help='Path to the Kaggle CSV file.')
    parser.add_argument('--output', default='kaggle_health_data_prepared.csv', help='Output CSV path.')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f'Input CSV not found: {args.input}')
    prepare_dataset(args.input, args.output)


if __name__ == '__main__':
    main()
