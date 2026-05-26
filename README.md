# EthioHealth-AI

EthioHealth-AI is an academic-grade clinical decision support system for Ethiopian healthcare. It includes disease classification, risk level prediction, length-of-stay regression, patient clustering, and explainable AI output.

## Project Structure

- `app.py` - Streamlit application for user interaction and prediction.
- `train.py` - Training pipeline for models and preprocessing artifacts.
- `utils.py` - Reusable helpers for preprocessing, encoding, evaluation, and explainability.
- `requirements.txt` - Python dependencies.
- `models/` - Saved trained models and artifacts.
- `ethiopia_health_data.csv` - Sample synthetic dataset file.

## Features

- Disease Prediction (classification)
- Risk Level Prediction (Low / Medium / High)
- Length of Stay Prediction (regression)
- Patient Clustering (K-Means, optional DBSCAN)
- Bagging ensemble classification for stronger disease and risk prediction
- Multi-task learning model for joint disease and risk prediction
- Explainable AI outputs via feature importance
- Ethiopian healthcare theme with Amharic labels

## Algorithms Used

EthioHealth-AI trains multiple machine learning algorithms because clinical decision support problems can behave differently depending on the dataset. Some models are strong for linear patterns, some are better for complex non-linear relationships, and some are useful for grouping similar patients. The training pipeline compares classification models and saves the best performing model for disease prediction and risk prediction.

### 1. Logistic Regression

Logistic Regression is used as a baseline classification model for disease prediction and risk level prediction.

Why it is used:

- It is simple, fast, and easy to interpret.
- It works well when the relationship between patient features and outcomes is close to linear.
- It gives a reliable baseline to compare more complex models against.

How it is implemented:

- In `train.py`, it is created with `LogisticRegression(max_iter=500, random_state=42)`.
- It is trained on preprocessed patient data.
- The model predicts encoded disease labels and encoded risk labels.

How to optimize it:

- Tune `C`, which controls regularization strength.
- Try different solvers such as `lbfgs`, `liblinear`, or `saga`.
- Scale numeric features correctly, which this project already does using `StandardScaler`.
- Check class imbalance and use `class_weight='balanced'` if one disease or risk group dominates the data.

### 2. Decision Tree Classifier

Decision Tree is used for disease prediction and risk level prediction.

Why it is used:

- It can learn non-linear clinical patterns.
- It is easier to explain than many black-box models.
- It can show which patient features are important for a decision.

How it is implemented:

- In `train.py`, it is created with `DecisionTreeClassifier(random_state=42)`.
- It is trained using the same processed feature matrix as the other classifiers.
- It is evaluated using accuracy, precision, recall, and F1-score.

How to optimize it:

- Tune `max_depth` to avoid overfitting.
- Tune `min_samples_split` and `min_samples_leaf` to make the tree more stable.
- Use pruning parameters such as `ccp_alpha`.
- Compare feature importance values to confirm the model is learning medically reasonable patterns.

### 3. Support Vector Machine (SVM)

SVM is used as another classification model for disease prediction and risk level prediction.

Why it is used:

- It can perform well on high-dimensional processed data.
- It is effective when the boundary between classes is complex.
- It can use probability estimates for confidence display in the app.

How it is implemented:

- In `train.py`, it is created with `SVC(probability=True, random_state=42)`.
- `probability=True` allows the Streamlit app to show prediction confidence.
- The model is trained and compared against the other classifiers.

How to optimize it:

- Tune `C` to control the margin and classification errors.
- Tune `kernel`, such as `linear`, `rbf`, or `poly`.
- Tune `gamma` when using the `rbf` kernel.
- Keep feature scaling enabled because SVM is sensitive to feature scale.

### 4. K-Nearest Neighbors (KNN)

KNN is used for disease prediction and risk level prediction.

Why it is used:

- It predicts based on similarity between patients.
- It is useful when patients with similar symptoms and vital signs have similar outcomes.
- It provides a simple comparison against more complex algorithms.

How it is implemented:

- In `train.py`, it is created with `KNeighborsClassifier()`.
- It uses the processed patient features to find nearby examples in the training data.
- It predicts the class most common among the nearest neighbors.

How to optimize it:

- Tune `n_neighbors`.
- Try different distance metrics such as `euclidean` or `manhattan`.
- Tune `weights`, using either `uniform` or `distance`.
- Keep numeric features scaled because distance-based models are affected by large feature ranges.

### 5. Multi-Layer Perceptron (MLP Neural Network)

MLP is used as a neural network classifier for disease prediction and risk level prediction.

Why it is used:

- It can learn more complex non-linear relationships.
- It is useful when symptom combinations interact in ways that simpler models may miss.
- It provides a stronger model option when the dataset is large enough.

How it is implemented:

- In `train.py`, it is created with `MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)`.
- The network has two hidden layers with 64 and 32 neurons.
- It is trained on the preprocessed patient feature matrix.

How to optimize it:

- Tune `hidden_layer_sizes`.
- Tune `learning_rate_init`.
- Tune `alpha`, which controls regularization.
- Increase `max_iter` if the model does not converge.
- Use more training data because neural networks usually improve with larger datasets.

### 6. Bagging Classifier

Bagging is an ensemble method used for disease prediction and risk level prediction.

Why it is used:

- It improves stability by training many decision trees on different random samples of the data.
- It reduces overfitting compared with using one decision tree alone.
- It is useful for clinical prediction because patient data can contain noise, missing patterns, and variation between cases.
- It can still support explainability because the feature importance values from the trees can be averaged.

How it is implemented:

- In `train.py`, it is created with `BaggingClassifier`.
- The base model is `DecisionTreeClassifier(max_depth=8, random_state=42)`.
- The ensemble trains `50` decision trees.
- Each tree trains on `80%` of the training samples using bootstrap sampling.
- The model is evaluated with the same accuracy, precision, recall, and F1-score metrics as the other classifiers.
- If Bagging gets the best weighted F1-score, it is saved as either `models/disease_model.joblib` or `models/risk_model.joblib`.

How to optimize it:

- Tune `n_estimators` to control how many trees are trained.
- Tune `max_samples` to control how much data each tree sees.
- Tune the base tree's `max_depth`, `min_samples_split`, and `min_samples_leaf`.
- Use cross-validation to choose settings that generalize well.
- Compare Bagging against the single Decision Tree to confirm the ensemble is improving performance.

How explainability works for Bagging:

- The project checks each tree inside the Bagging model.
- It collects each tree's `feature_importances_`.
- It averages those values to show the most influential patient features in the Streamlit app.

### 7. Multi-Task Learning

Multi-task learning is used to train one model structure to predict more than one related clinical output. In this project, the multi-task model predicts both `Disease` and `Risk_Level` from the same patient profile.

Why it is used:

- Disease prediction and risk level prediction are related clinical tasks.
- A patient profile that suggests a certain disease may also provide useful signals about clinical risk.
- Training both outputs together helps the system evaluate these related targets in one model workflow.
- It creates a useful comparison against training two completely separate models.

How it is implemented:

- In `train.py`, it is implemented with `MultiOutputClassifier`.
- The base estimator is `RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)`.
- The target values `Disease` and `Risk_Level` are joined into one multi-output target matrix.
- The model learns to predict both outputs from the same preprocessed input features.
- It is saved as `models/multi_task_model.joblib`.
- The current Streamlit app still uses the best separate disease and risk models for the main prediction buttons, while the multi-task model is saved for comparison, experimentation, and future combined prediction workflows.

How to optimize it:

- Tune the base `RandomForestClassifier` parameters such as `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`.
- Use cross-validation to compare the multi-task model against the separate disease and risk models.
- Compare the F1-score for both outputs, not only one target.
- Check whether one task improves while the other becomes worse.
- Use larger real clinical datasets because multi-task learning is more useful when shared patterns exist across related targets.

### 8. Random Forest Regressor

Random Forest Regressor is used for length-of-stay prediction.

Why it is used:

- Length of stay is a regression problem because the output is a number of days.
- Random Forest can model non-linear relationships between patient condition and hospital stay.
- It is robust and usually performs well without heavy tuning.

How it is implemented:

- In `train.py`, it is created with `RandomForestRegressor(n_estimators=150, random_state=42)`.
- It predicts the `Length_of_Stay` column.
- It is evaluated using MSE, RMSE, MAE, and R2 score.

How to optimize it:

- Tune `n_estimators` for the number of trees.
- Tune `max_depth` to control model complexity.
- Tune `min_samples_split` and `min_samples_leaf`.
- Use MAE and RMSE together to understand average error and large-error behavior.
- Check whether extreme length-of-stay values are outliers or real cases.

### 9. K-Means Clustering

K-Means is used for patient clustering.

Why it is used:

- It groups patients with similar profiles.
- It can help identify common patient types or risk groups.
- It supports exploratory clinical analysis even when no target label is used.

How it is implemented:

- In `train.py`, it is created with `KMeans(n_clusters=4, random_state=42, n_init=10)`.
- It is trained on all processed patient features.
- The project prints the number of patients in each cluster after training.

How to optimize it:

- Tune `n_clusters`.
- Use the elbow method to compare cluster compactness.
- Use silhouette score to measure cluster separation.
- Review cluster profiles with clinicians to ensure the groups are meaningful.

### 10. DBSCAN

DBSCAN is used as an optional clustering model.

Why it is used:

- It can detect unusual or isolated patient patterns.
- It does not require choosing the number of clusters in advance.
- It can mark noisy records as outliers.

How it is implemented:

- In `train.py`, it is created with `DBSCAN(eps=0.75, min_samples=8)`.
- It is trained on the processed patient features.
- The project prints the unique cluster labels found by DBSCAN.

How to optimize it:

- Tune `eps`, which controls how close points must be to form a cluster.
- Tune `min_samples`, which controls the minimum number of nearby points required.
- Use nearest-neighbor distance plots to choose a better `eps`.
- Review outliers carefully because they may represent rare but important clinical cases.

## Model Implementation Workflow

The model training process is implemented in `train.py`:

1. Load the CSV dataset using `load_dataset`.
2. Validate that all required columns exist.
3. Separate input features from target columns.
4. Encode target labels for `Disease` and `Risk_Level`.
5. Preprocess numeric and categorical features.
6. Train multiple classification models for disease prediction, including Bagging.
7. Select the best disease model using weighted F1-score.
8. Train multiple classification models for risk level prediction, including Bagging.
9. Select the best risk model using weighted F1-score.
10. Train a multi-task model that predicts both disease and risk level.
11. Train the Random Forest regression model for length of stay.
12. Train K-Means and DBSCAN clustering models.
13. Save all trained models and preprocessing artifacts into the `models/` folder.

## Explainable AI

The application includes explainability support through `utils.py`.

How explainability works:

- For tree-based models, the app uses `feature_importances_`.
- For Bagging models, the app averages feature importance values from the trees inside the ensemble.
- For linear models, the app uses coefficient values from `coef_`.
- The `explain_prediction` function ranks the most important features.
- The Streamlit app displays the top features that influenced a prediction.

Why explainability is important:

- Clinicians need to understand why a model suggested a result.
- Explanations help detect incorrect or biased model behavior.
- Feature importance can show whether the model is relying on clinically meaningful inputs.

Current limitation:

- Some models, such as SVM, KNN, and MLP, do not expose simple built-in feature importance in the same way as trees or linear models.
- For stronger explainability, future versions can add SHAP or LIME.

## Model Optimization Strategy

To improve model performance, use these steps:

1. Improve data quality.
   Clean missing values, incorrect labels, duplicate records, and unrealistic vital signs.

2. Balance the dataset.
   If some diseases or risk levels appear much more often than others, use class balancing, resampling, or more representative data.

3. Tune hyperparameters.
   Use `GridSearchCV` or `RandomizedSearchCV` to test parameter combinations for each algorithm.

4. Use cross-validation.
   Cross-validation gives a more reliable estimate than a single train-test split.

5. Compare multiple metrics.
   For classification, compare accuracy, precision, recall, and F1-score. In clinical systems, recall can be very important because missing high-risk patients may be dangerous.

6. Evaluate regression error clearly.
   For length-of-stay prediction, compare MAE, RMSE, and R2. MAE is easy to understand because it shows the average error in days.

7. Review model explanations.
   Check whether important features make clinical sense. If explanations look wrong, review the data and feature engineering.

8. Validate with real clinical data.
   The included dataset is synthetic. Before real use, train and test the system with approved Ethiopian hospital data.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Training

Train models using your dataset with the required columns:

- `Age`
- `Gender`
- `Region`
- `Fever`
- `Cough`
- `Headache`
- `Fatigue`
- `Vomiting`
- `Diarrhea`
- `Temperature`
- `Heart Rate`
- `Comorbidity`
- `Season`
- `Disease`
- `Risk_Level`
- `Length_of_Stay`

Run:

```bash
python train.py --data ethiopia_health_data.csv
```

If your dataset file is named differently, pass it with `--data`:

```bash
python train.py --data /path/to/your_dataset.csv
```

## Using a Kaggle Dataset

Recommended Kaggle dataset:

- **Disease Diagnosis Dataset**
- Link: https://www.kaggle.com/datasets/s3programmer/disease-diagnosis-dataset

Why this dataset is suitable:

- It contains patient age and gender.
- It contains symptoms.
- It contains heart rate and body temperature.
- It contains diagnosis labels that can be used as `Disease`.
- It contains severity labels that can be mapped to `Risk_Level`.
- It is appropriate for disease prediction and clinical risk classification experiments.

The app requires a specific training schema, so this project includes a converter:

```text
prepare_kaggle_disease_diagnosis.py
```

### Download with Kaggle API

Install the Kaggle API:

```bash
pip install kaggle
```

Create a Kaggle API token:

1. Open your Kaggle account.
2. Go to `Settings`.
3. Go to the `API` section.
4. Click `Create New Token`.
5. Save the downloaded `kaggle.json` file.

Place the token in:

```text
~/.kaggle/kaggle.json
```

Then run:

```bash
kaggle datasets download -d s3programmer/disease-diagnosis-dataset --unzip
```

After downloading, find the CSV file name, then convert it to the app schema:

```bash
python prepare_kaggle_disease_diagnosis.py --input downloaded_file.csv --output kaggle_health_data_prepared.csv
```

Train the models with the prepared Kaggle dataset:

```bash
python train.py --data kaggle_health_data_prepared.csv
```

Then run the app:

```bash
streamlit run app.py
```

### Kaggle Dataset Mapping

The converter maps the Kaggle dataset into the columns required by this app:

- `Age` comes from the Kaggle age column.
- `Gender` comes from the Kaggle gender/sex column.
- `Fever`, `Cough`, `Headache`, `Fatigue`, `Vomiting`, and `Diarrhea` are detected from symptom text columns.
- `Temperature` comes from body temperature.
- `Heart Rate` comes from heart rate.
- `Disease` comes from diagnosis.
- `Risk_Level` comes from severity.
- `Length_of_Stay` is estimated from severity, treatment plan, temperature, and heart rate because this Kaggle dataset does not provide a real length-of-stay field.

For real hospital deployment, replace the estimated `Length_of_Stay` with actual admission/discharge or stay-duration data.

This command saves:

- `models/preprocessor.joblib`
- `models/disease_model.joblib`
- `models/risk_model.joblib`
- `models/multi_task_model.joblib`
- `models/stay_model.joblib`
- `models/disease_label_encoder.joblib`
- `models/risk_label_encoder.joblib`
- `models/kmeans_model.joblib`
- `models/dbscan_model.joblib`
- `models/feature_names.joblib`

## Running the Streamlit App

Start the app with:

```bash
streamlit run app.py
```

Use the sidebar to enter patient details, then click the prediction buttons:

- `Predict Disease / የበሽታ እትንት`
- `Predict Risk / የአደጋ ደረጃ እትንት`
- `Predict Stay / የማረፊያ ጊዜ እትንት`

## Deployment

This project is a Streamlit application, so the easiest deployment option is Streamlit Community Cloud. You can also deploy it on Hugging Face Spaces, Render, Railway, or another Python web hosting platform.

### Recommended Option: Streamlit Community Cloud

Use this option for a simple public demo or academic project.

Before deploying, make sure these files and folders are committed to GitHub:

- `app.py`
- `requirements.txt`
- `utils.py`
- `models/`
- `.streamlit/config.toml`

Deployment steps:

1. Create a GitHub repository.
2. Push this project to the repository.
3. Go to Streamlit Community Cloud.
4. Choose `Create app`.
5. Select your GitHub repository.
6. Set the main file path to:

```text
app.py
```

7. Choose a Python version supported by Streamlit Cloud.
8. Click `Deploy`.

Streamlit Cloud installs the packages from `requirements.txt`, then runs the app with:

```bash
streamlit run app.py
```

### Alternative Option: Hugging Face Spaces

Use this option if you want a machine learning demo page that is easy to share.

Steps:

1. Create a new Hugging Face Space.
2. Select `Streamlit` as the SDK.
3. Upload or push this project into the Space repository.
4. Make sure `requirements.txt` is included.
5. Make sure the saved model files are included in the `models/` folder.

Hugging Face Spaces will install dependencies and run the Streamlit app automatically.

### Alternative Option: Render

Use this option if you want more control over the deployment environment.

Recommended Render settings:

- Service type: `Web Service`
- Runtime: `Python`
- Build command:

```bash
pip install -r requirements.txt
```

- Start command:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

### Important Deployment Notes

- Do not deploy `__pycache__/` files.
- Keep `.streamlit/secrets.toml` private if you ever add secrets.
- The `models/` folder must be included because the app loads trained `.joblib` files at runtime.
- If model files become too large for normal GitHub commits, use Git LFS.
- The current dataset is synthetic. Do not deploy real patient data unless it is approved, anonymized, and compliant with your institution's rules.

### Quick Deployment Checklist

- App runs locally with `streamlit run app.py`.
- `requirements.txt` contains all dependencies.
- `models/` contains the trained model files.
- GitHub repository contains the latest code.
- Deployment platform points to `app.py`.

## Notes

- Ensure the `models/` directory contains the saved artifacts before launching `app.py`.
- The sample dataset provided is synthetic and intended for testing only.
- For production use, replace the synthetic dataset with your real Ethiopian hospital data.

## Troubleshooting

- If `train.py` cannot find your dataset, verify the file path and the `--data` argument.
- If `app.py` reports missing model files, rerun `train.py` and confirm `models/` contains the artifacts.

## Contact

This repository is designed to support Ethiopian healthcare teams with academic-quality AI decision assistance.
