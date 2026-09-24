# Maternal Health Risk Prediction using Machine Learning

## Overview

This project uses machine learning to classify maternal health records into **Low Risk** or **High Risk** based on clinical and demographic indicators.

The project follows an end-to-end machine learning workflow:

**Real-world problem → Data exploration → Feature analysis → Preprocessing → Model training → Evaluation → Model comparison → Prediction**

Three classification algorithms are trained and evaluated:

* Support Vector Machine (SVM)
* Decision Tree
* Random Forest

The project also includes exploratory data analysis, correlation analysis, confusion matrices, ROC-AUC analysis, feature importance, and model serialization for future inference.

> **Disclaimer:** This project is intended for educational and machine-learning demonstration purposes only. It is not a medical diagnostic tool and should not be used to make clinical decisions.

---

## Project Objectives

The main objectives are to:

1. Explore maternal health data and identify patterns.
2. Analyze relationships between clinical variables and maternal risk.
3. Prepare the dataset for machine learning.
4. Train multiple classification models.
5. Compare model performance using relevant classification metrics.
6. Analyze which health indicators contribute most to predictions.
7. Save the trained model for inference on new patient records.

---

## Dataset

The model uses a CSV dataset named:

```text
Dataset - Updated.csv
```

The target variable is:

```text
Risk Level
```

Target encoding:

| Risk Level | Encoded Value |
| ---------- | ------------: |
| Low        |             0 |
| High       |             1 |

The project uses clinical and demographic features such as:

* Age
* Systolic Blood Pressure
* Diastolic Blood Pressure
* Blood Sugar
* Body Temperature
* BMI
* Previous Complications
* Preexisting Diabetes
* Gestational Diabetes
* Mental Health
* Heart Rate

---

## Machine Learning Workflow

### 1. Data Loading

The dataset is loaded using Pandas and its dimensions, data types, missing values, and sample records are inspected.

### 2. Exploratory Data Analysis

The notebook investigates:

* Target-class distribution
* Average blood sugar across risk groups
* Systolic vs. diastolic blood pressure
* Age vs. blood sugar
* Descriptive statistics

### 3. Correlation Analysis

The project converts the target into a numerical representation:

```text
Low Risk  → 0
High Risk → 1
```

A correlation matrix and heatmap are then used to investigate relationships between health metrics and risk.

### 4. Data Preparation

The preprocessing pipeline performs:

* Removal of records with missing target labels
* Separation of features and target
* 80/20 train-test split
* Stratified sampling
* Median imputation for missing numerical values
* Standardization using `StandardScaler`

The train/test split uses:

```python
random_state = 42
```

to make the experiment reproducible.

### 5. Model Training

Three classification algorithms are trained:

#### Support Vector Machine

```python
SVC(kernel='rbf', random_state=42)
```

#### Decision Tree

```python
DecisionTreeClassifier(random_state=42)
```

#### Random Forest

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

### 6. Model Evaluation

The models are compared using:

* Accuracy
* Precision
* Recall
* F1-score
* Specificity
* Confusion Matrix
* ROC Curve
* AUC

For this project, **High Risk is treated as the positive class**.

This makes recall particularly important when examining how effectively a model identifies records classified as High Risk.

### 7. Feature Importance

Random Forest feature importance is used to examine which health indicators contribute most strongly to the model's predictions.

### 8. Model Serialization

The trained Random Forest model and fitted scaler are saved using `joblib`:

```text
maternal_risk_rf_model.pkl
scaler.pkl
```

These files allow the preprocessing and trained model to be reused without retraining.

---

## Example Inference

The notebook contains an inference function:

```python
assess_patient_risk(patient_vitals)
```

A new patient can be represented as a Python dictionary:

```python
sample_patient = {
    'Age': 35,
    'Systolic BP': 140,
    'Diastolic': 95,
    'BS': 11.0,
    'Body Temp': 98,
    'BMI': 31.5,
    'Previous Complications': 1,
    'Preexisting Diabetes': 1,
    'Gestational Diabetes': 0,
    'Mental Health': 0,
    'Heart Rate': 88
}
```

The saved scaler transforms the input features before the saved Random Forest model generates a prediction.

The inference output includes:

* Predicted risk class
* Model probability for Low Risk
* Model probability for High Risk
* Predicted-class probability

---

## Project Outputs

The notebook generates analytical outputs including:

### Data Analysis

```text
1_descriptive_statistics.csv
2_correlation_matrix.csv
```

### Model Evaluation

```text
3_model_comparison_leaderboard.csv
```

### Visualizations

* EDA pattern overview
* Correlation heatmap
* Model performance comparison
* Confusion matrices
* ROC curves
* Decision tree visualization
* Random Forest feature importance

These files are stored in the `outputs/` directory.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/maternal-health-risk-prediction.git
cd maternal-health-risk-prediction
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
mlproject.ipynb
```

Make sure the following files are in the same project directory:

```text
mlproject.ipynb
Dataset - Updated.csv
```

Run the notebook cells sequentially.

---

## Project Structure

```text
maternal-health-risk-prediction/
│
├── mlproject.ipynb
├── Dataset - Updated.csv
├── requirements.txt
├── README.md
│
├── maternal_risk_rf_model.pkl
├── scaler.pkl
│
└── outputs/
    ├── descriptive statistics
    ├── correlation analysis
    ├── model metrics
    └── visualization outputs
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib
* Jupyter Notebook

---

## Key Machine Learning Concepts Demonstrated

* Exploratory Data Analysis
* Classification
* Train-Test Splitting
* Stratified Sampling
* Missing-Value Imputation
* Feature Scaling
* Support Vector Machines
* Decision Trees
* Random Forest
* Confusion Matrix
* Precision
* Recall
* F1-score
* Specificity
* ROC-AUC
* Feature Importance
* Model Serialization
* Model Inference

---

## Limitations

This project has several limitations:

* The dataset may not represent the full diversity of real-world maternal populations.
* Model performance depends heavily on the dataset and feature quality.
* A single train-test split is used rather than cross-validation.
* Hyperparameter optimization has not been extensively performed.
* Feature importance from Random Forest should not be interpreted as causal relationships.
* Model probabilities should not be interpreted as clinical certainty.
* External clinical validation is not performed.

Therefore, the model should be considered a **machine-learning prototype rather than a validated clinical system**.

---

## Future Improvements

Potential extensions include:

1. Cross-validation for more robust performance estimation.
2. Hyperparameter optimization using GridSearchCV or RandomizedSearchCV.
3. Calibration of predicted probabilities.
4. Additional classification algorithms such as Logistic Regression, XGBoost, or LightGBM.
5. Explainable AI using SHAP.
6. Feature engineering based on domain knowledge.
7. External validation on an independent dataset.
8. Model monitoring and data-drift detection.
9. Deployment through an API or interactive dashboard.
10. Evaluation using clinically relevant decision thresholds.

---

## Author

**Vithiyasagar**

Computer Science Engineering
NIT Puducherry

---

## Disclaimer

This repository is intended for **educational and research purposes only**.

The predictions produced by this project must not be used as a substitute for professional medical advice, diagnosis, or treatment.
