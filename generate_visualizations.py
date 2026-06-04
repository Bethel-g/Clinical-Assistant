import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Create assets directory if it doesn't exist
os.makedirs('assets/visualizations', exist_ok=True)

def load_and_merge_datasets():
    """Load and merge all available datasets"""
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df3 = pd.DataFrame()
    
    if os.path.exists('ethiopian_hospital_dataset.xlsx'):
        df1 = pd.read_excel('ethiopian_hospital_dataset.xlsx')
        df1 = df1.rename(columns={column: column.replace('_', ' ') for column in df1.columns})
        df1 = df1.rename(columns={'Risk Level': 'Risk_Level', 'Length of Stay': 'Length_of_Stay'})
        
    if os.path.exists('clinical_dataset.csv'):
        df2 = pd.read_csv('clinical_dataset.csv')
        rename_map = {
            'sex': 'Gender', 'age': 'Age', 'temperature': 'Temperature',
            'pulse': 'Heart Rate', 'target': 'Disease', 'weight': 'Weight',
            'height': 'Height', 'bmi': 'BMI', 'oxygen_saturation': 'Oxygen Saturation',
            'blood_pressure_systolic': 'Blood Pressure Systolic',
            'blood_pressure_diastolic': 'Blood Pressure Diastolic', 'pain_score': 'Pain Score'
        }
        df2 = df2.rename(columns=rename_map)
        
    if os.path.exists('clinical_dataset.xlsx'):
        df3 = pd.read_excel('clinical_dataset.xlsx')
        df3 = df3.rename(columns=rename_map)

    for df_new in [df2, df3]:
        if 'Gender' in df_new.columns:
            df_new['Gender'] = df_new['Gender'].str.capitalize()
            
    df = pd.concat([df1, df2, df3], ignore_index=True).drop_duplicates()
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def prepare_test_data(df):
    """Prepare test data matching training pipeline"""
    from utils import encode_labels
    
    expected_features = [
        'Age', 'Gender', 'Region', 'Fever', 'Cough', 'Headache', 'Fatigue',
        'Vomiting', 'Diarrhea', 'Chest Pain', 'Shortness of Breath', 'Dizziness',
        'Temperature', 'Heart Rate', 'WBC Count', 'Hemoglobin', 'Malaria Test',
        'Comorbidity', 'Season', 'Weight', 'Height', 'BMI', 'Oxygen Saturation', 
        'Blood Pressure Systolic', 'Blood Pressure Diastolic', 'Pain Score'
    ]
    
    default_values = {
        'Chest Pain': 'No', 'Shortness of Breath': 'No', 'Dizziness': 'No',
        'WBC Count': np.nan, 'Hemoglobin': np.nan, 'Malaria Test': 'Unknown',
        'Weight': np.nan, 'Height': np.nan, 'BMI': np.nan,
        'Oxygen Saturation': np.nan, 'Blood Pressure Systolic': np.nan,
        'Blood Pressure Diastolic': np.nan, 'Pain Score': np.nan,
        'Disease': 'Unknown', 'Risk_Level': 'Low', 'Length_of_Stay': 0,
        'Gender': 'Unknown', 'Region': 'Unknown', 'Fever': 'Unknown',
        'Cough': 'Unknown', 'Headache': 'Unknown', 'Fatigue': 'Unknown',
        'Vomiting': 'Unknown', 'Diarrhea': 'Unknown', 'Comorbidity': 'Unknown', 'Season': 'Unknown'
    }
    
    for col, default in default_values.items():
        if col not in df.columns:
            df[col] = default

    df['Disease'] = df['Disease'].fillna('Unknown')
    df['Risk_Level'] = df['Risk_Level'].fillna('Low')
    
    disease_counts = df['Disease'].value_counts()
    rare_diseases = disease_counts[disease_counts < 50].index
    df.loc[df['Disease'].isin(rare_diseases), 'Disease'] = 'Other'

    X = df[expected_features].copy()
    y_disease, disease_encoder = encode_labels(df, 'Disease')
    y_risk, risk_encoder = encode_labels(df, 'Risk_Level')
    y_stay = df['Length_of_Stay'].replace({np.nan: 0}).astype(float)
    
    numeric_features = [
        'Age', 'Temperature', 'Heart Rate', 'WBC Count', 'Hemoglobin',
        'Weight', 'Height', 'BMI', 'Oxygen Saturation', 'Blood Pressure Systolic',
        'Blood Pressure Diastolic', 'Pain Score'
    ]
    categorical_features = [col for col in expected_features if col not in numeric_features]
    
    return X, y_disease, y_risk, y_stay, disease_encoder, risk_encoder, numeric_features, categorical_features, df

def plot_disease_confusion_matrix(y_true, y_pred, disease_labels):
    """Plot confusion matrix for disease prediction"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=disease_labels, yticklabels=disease_labels, cbar=True)
    plt.title('Disease Prediction - Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('assets/visualizations/01_disease_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 01_disease_confusion_matrix.png")
    plt.close()

def plot_risk_confusion_matrix(y_true, y_pred, risk_labels):
    """Plot confusion matrix for risk prediction"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn_r', xticklabels=risk_labels, yticklabels=risk_labels, 
                cbar_kws={'label': 'Count'}, annot_kws={'size': 14})
    plt.title('Risk Level Prediction - Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Risk Level', fontsize=12)
    plt.ylabel('Actual Risk Level', fontsize=12)
    plt.tight_layout()
    plt.savefig('assets/visualizations/02_risk_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 02_risk_confusion_matrix.png")
    plt.close()

def plot_model_performance():
    """Plot overall model performance metrics"""
    models = ['Disease\nPrediction', 'Risk Level\nPrediction', 'LOS\nPrediction (R²)']
    accuracy_scores = [0.0134, 0.3895, 0.2261]
    colors = ['#FF6B6B', '#FFD93D', '#6BCB77']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, accuracy_scores, color=colors, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, score in zip(bars, accuracy_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2%}' if score < 1 else f'{score:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Overview', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(accuracy_scores) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('assets/visualizations/03_model_performance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 03_model_performance.png")
    plt.close()

def plot_risk_metrics():
    """Plot risk model detailed metrics"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    scores = [0.3895, 0.3960, 0.3853, 0.3839]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metrics, scores, color='#FFD93D', edgecolor='black', linewidth=2)
    
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Risk Level Prediction - Detailed Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 0.5)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('assets/visualizations/04_risk_metrics.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 04_risk_metrics.png")
    plt.close()

def plot_los_regression_metrics():
    """Plot LOS regression metrics"""
    metrics = ['MAE', 'RMSE', 'R² Score']
    values = [3.1963, 3.8516, 0.2261]
    colors_los = ['#FF6B6B', '#FF8C42', '#6BCB77']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metrics, values, color=colors_los, edgecolor='black', linewidth=2)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title('Length of Stay (LOS) - Regression Metrics', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('assets/visualizations/05_los_metrics.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 05_los_metrics.png")
    plt.close()

def plot_data_distribution(df):
    """Plot data distribution analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Age distribution
    axes[0, 0].hist(df['Age'].dropna(), bins=30, color='#6BCB77', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Age Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('Age (years)')
    axes[0, 0].set_ylabel('Frequency')
    
    # Gender distribution
    gender_counts = df['Gender'].value_counts()
    axes[0, 1].bar(gender_counts.index, gender_counts.values, color=['#FF6B6B', '#4ECDC4', '#FFD93D'], edgecolor='black')
    axes[0, 1].set_title('Gender Distribution', fontweight='bold')
    axes[0, 1].set_ylabel('Count')
    for i, v in enumerate(gender_counts.values):
        axes[0, 1].text(i, v + 100, str(v), ha='center', fontweight='bold')
    
    # Temperature distribution
    axes[1, 0].hist(df['Temperature'].dropna(), bins=25, color='#FF8C42', edgecolor='black', alpha=0.7)
    axes[1, 0].set_title('Temperature Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('Temperature (°C)')
    axes[1, 0].set_ylabel('Frequency')
    
    # Length of Stay distribution
    axes[1, 1].hist(df['Length_of_Stay'].dropna(), bins=15, color='#A8E6CF', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('Length of Stay Distribution', fontweight='bold')
    axes[1, 1].set_xlabel('Days')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('assets/visualizations/06_data_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 06_data_distribution.png")
    plt.close()

def plot_disease_distribution(df):
    """Plot disease distribution"""
    disease_counts = df['Disease'].value_counts().head(15)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    disease_counts.plot(kind='barh', ax=ax, color='#6BCB77', edgecolor='black')
    ax.set_title('Top 15 Disease Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Count')
    ax.invert_yaxis()
    
    # Add value labels
    for i, v in enumerate(disease_counts.values):
        ax.text(v + 50, i, str(v), va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('assets/visualizations/07_disease_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 07_disease_distribution.png")
    plt.close()

def plot_risk_distribution(df):
    """Plot risk level distribution"""
    risk_counts = df['Risk_Level'].value_counts()
    colors_risk = {'High': '#FF6B6B', 'Medium': '#FFD93D', 'Low': '#6BCB77'}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(risk_counts.index, risk_counts.values, 
                   color=[colors_risk.get(x, '#999') for x in risk_counts.index],
                   edgecolor='black', linewidth=2)
    
    ax.set_title('Patient Risk Level Distribution', fontsize=14, fontweight='bold')
    ax.set_ylabel('Patient Count')
    ax.set_ylabel('Number of Patients')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('assets/visualizations/08_risk_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 08_risk_distribution.png")
    plt.close()

def plot_los_analysis(df):
    """Plot LOS analysis by risk level"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data_to_plot = [df[df['Risk_Level'] == risk]['Length_of_Stay'].dropna().values 
                    for risk in ['Low', 'Medium', 'High']]
    
    bp = ax.boxplot(data_to_plot, labels=['Low', 'Medium', 'High'], patch_artist=True)
    
    colors = ['#6BCB77', '#FFD93D', '#FF6B6B']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_title('Length of Stay by Risk Level', fontsize=14, fontweight='bold')
    ax.set_ylabel('Length of Stay (days)')
    ax.set_xlabel('Risk Level')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('assets/visualizations/09_los_by_risk.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 09_los_by_risk.png")
    plt.close()

def plot_vital_stats(df):
    """Plot vital statistics"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    vitals = [
        ('Heart Rate', axes[0, 0], '#FF6B6B'),
        ('Oxygen Saturation', axes[0, 1], '#6BCB77'),
        ('BMI', axes[1, 0], '#FFD93D'),
        ('Hemoglobin', axes[1, 1], '#4ECDC4')
    ]
    
    for vital, ax, color in vitals:
        if vital in df.columns:
            data = df[vital].dropna()
            ax.hist(data, bins=25, color=color, edgecolor='black', alpha=0.7)
            ax.set_title(f'{vital} Distribution', fontweight='bold')
            ax.set_xlabel(vital)
            ax.set_ylabel('Frequency')
            ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('assets/visualizations/10_vital_stats.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 10_vital_stats.png")
    plt.close()

def generate_visualizations():
    """Main visualization generation function"""
    print("\n" + "="*80)
    print("📊 GENERATING MODEL PERFORMANCE & DATA ANALYSIS VISUALIZATIONS")
    print("="*80 + "\n")
    
    # Load data
    print("📦 Loading data...")
    df = load_and_merge_datasets()
    X, y_disease, y_risk, y_stay, disease_enc, risk_enc, numeric_feat, categorical_feat, df_full = prepare_test_data(df)
    
    # Split data
    X_train, X_test, y_d_train, y_d_test, y_r_train, y_r_test = train_test_split(
        X, y_disease, y_risk, test_size=0.2, random_state=42
    )
    X_train_stay, X_test_stay, y_s_train, y_s_test = train_test_split(
        X, y_stay, test_size=0.2, random_state=42
    )
    
    # Load models
    print("📦 Loading models...")
    disease_model = joblib.load('models/disease_model.joblib')
    risk_model = joblib.load('models/risk_model.joblib')
    stay_model = joblib.load('models/stay_model.joblib')
    
    # Generate predictions
    print("🔮 Generating predictions...")
    y_d_pred = disease_model.predict(X_test)
    y_r_pred = risk_model.predict(X_test)
    
    # Generate visualizations
    print("\n📈 Creating visualizations...\n")
    
    plot_disease_confusion_matrix(y_d_test, y_d_pred, disease_enc.classes_)
    plot_risk_confusion_matrix(y_r_test, y_r_pred, risk_enc.classes_)
    plot_model_performance()
    plot_risk_metrics()
    plot_los_regression_metrics()
    plot_data_distribution(df)
    plot_disease_distribution(df)
    plot_risk_distribution(df)
    plot_los_analysis(df)
    plot_vital_stats(df)
    
    print("\n" + "="*80)
    print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
    print("📁 Location: assets/visualizations/")
    print("="*80 + "\n")

if __name__ == "__main__":
    generate_visualizations()
