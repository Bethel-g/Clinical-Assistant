import pandas as pd
import os

df1 = pd.read_excel('ethiopian_hospital_dataset.xlsx') if os.path.exists('ethiopian_hospital_dataset.xlsx') else pd.DataFrame()
df2 = pd.read_csv('clinical_dataset.csv') if os.path.exists('clinical_dataset.csv') else pd.DataFrame()
df3 = pd.read_excel('clinical_dataset.xlsx') if os.path.exists('clinical_dataset.xlsx') else pd.DataFrame()

# Rename new dataset columns to match old where possible
rename_map = {
    'sex': 'Gender',
    'age': 'Age',
    'temperature': 'Temperature',
    'pulse': 'Heart Rate',
    'target': 'Disease',
    'weight': 'Weight',
    'height': 'Height',
    'bmi': 'BMI',
    'oxygen_saturation': 'Oxygen Saturation',
    'blood_pressure_systolic': 'Blood Pressure Systolic',
    'blood_pressure_diastolic': 'Blood Pressure Diastolic',
    'pain_score': 'Pain Score'
}
df2 = df2.rename(columns=rename_map)
df3 = df3.rename(columns=rename_map)

# Capitalize Gender in df2 and df3
for df_new in [df2, df3]:
    if 'Gender' in df_new.columns:
        df_new['Gender'] = df_new['Gender'].str.capitalize()

# Combine all
df = pd.concat([df1, df2, df3], ignore_index=True).drop_duplicates()

print("Combined dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
