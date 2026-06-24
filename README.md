# Loan Approval Prediction System — MLOps Project

End-to-end machine learning system that predicts loan approval (Y/N) based on applicant information, built with **scikit-learn**, **Flask**, **PostgreSQL**, and **MLflow**.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Pipeline Architecture](#pipeline-architecture)
- [Setup & Installation](#setup--installation)
- [Configuration](#configuration)
- [How to Run](#how-to-run)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [MLflow Model Registry](#mlflow-model-registry)
- [Web Interface](#web-interface)
- [API Endpoints](#api-endpoints)
- [Models Supported](#models-supported)
- [Feature Engineering](#feature-engineering)
- [Model Evaluation](#model-evaluation)
- [Logging & Error Handling](#logging--error-handling)
- [Deployment Notes](#deployment-notes)
- [Future Improvements](#future-improvements)

---

## Overview

This project implements a production-ready MLOps pipeline for **binary classification** of loan applications:

- **Input**: 11 applicant features (demographics + financial)
- **Output**: "Y" (approved) or "N" (rejected)
- **Data source**: PostgreSQL database (`loan_data` table)
- **Training**: Automated pipeline with data validation, feature engineering, SMOTE balancing, and hyperparameter-tuned models
- **Tracking**: Every run is logged to MLflow with params, metrics, artifacts, and model registry
- **Serving**: Flask web application for real-time predictions

---

## Dataset

The dataset is the classic **Kaggle Loan Prediction** problem, stored in PostgreSQL:

| Column | Type | Description |
|--------|------|-------------|
| `Loan_ID` | string | Unique identifier (dropped during training) |
| `Gender` | string | Male / Female |
| `Married` | string | Yes / No |
| `Dependents` | string | 0 / 1 / 2 / 3+ |
| `Education` | string | Graduate / Not Graduate |
| `Self_Employed` | string | Yes / No |
| `ApplicantIncome` | float | Applicant's income |
| `CoapplicantIncome` | float | Co-applicant's income |
| `LoanAmount` | float | Loan amount requested |
| `Loan_Amount_Term` | float | Loan repayment term (months) |
| `Credit_History` | float | 1.0 (has history) / 0.0 (no history) |
| `Property_Area` | string | Urban / Semiurban / Rural |
| `Loan_Status` | string | **Y** (approved) / **N** (rejected) — target |

- **Rows**: 614
- **Features**: 12 input columns
- **Target**: `Loan_Status` (binary: Y/N)

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.14 |
| ML Framework | scikit-learn, imbalanced-learn |
| Web Framework | Flask, gunicorn |
| Database | PostgreSQL (source), SQLite (MLflow) |
| Experiment Tracking | MLflow 3.12.0 |
| Model Registry | MLflow Model Registry (alias-based) |
| Data Handling | pandas, numpy |
| Serialization | pickle, PyYAML |
| Visualization | matplotlib, seaborn |
| Environment | python-dotenv |

---

## Project Structure

```
MLOps/
├── .env                          # PostgreSQL credentials
├── .gitignore
├── README.md
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
├── demo.py                       # Entry point — run training pipeline
├── app.py                        # Entry point — run Flask web server
├── mlflow.db                     # MLflow tracking database (auto-generated)
│
├── config/
│   ├── schema.yaml               # Column schema, types, domain values
│   └── model.yaml                # Model selection & hyperparameters
│
├── src/
│   ├── __init__.py
│   │
│   ├── constants/
│   │   └── __init__.py           # All configuration constants
│   │
│   ├── logger/
│   │   └── __init__.py           # Rotating file + colored console logger
│   │
│   ├── exception/
│   │   └── __init__.py           # Custom MyException with traceback
│   │
│   ├── configuration/
│   │   └── postgres_connection.py # PostgreSQLClient (psycopg2 + SQLAlchemy)
│   │
│   ├── data_access/
│   │   └── proj1_data.py         # Fetches loan_data table as DataFrame
│   │
│   ├── entity/
│   │   ├── config_entity.py      # Dataclasses: *Config for each pipeline stage
│   │   ├── artifact_entity.py    # Dataclasses: *Artifact for each pipeline stage
│   │   ├── estimator.py          # TargetValueMapping (Y→1, N→0)
│   │   └── local_estimator.py    # Local filesystem model registry
│   │
│   ├── utils/
│   │   └── main_utils.py         # YAML, pickle, numpy I/O utilities
│   │
│   ├── components/
│   │   ├── data_ingestion.py     # Fetch from PostgreSQL → train/test split
│   │   ├── data_validation.py    # Schema validation against config/schema.yaml
│   │   ├── data_transformation.py# Feature engineering + sklearn preprocessing
│   │   ├── model_trainer.py      # SMOTE + model training + threshold tuning
│   │   ├── model_evaluation.py   # Compare with existing model + MLflow registry
│   │   └── model_pusher.py       # Push accepted model to local registry
│   │
│   ├── pipline/                  # (typo: should be pipeline)
│   │   ├── training_pipeline.py  # Orchestrates all pipeline stages
│   │   └── prediction_pipeline.py# Loads model & preprocessor → predicts
│   │
│   └── monitoring/               # (placeholder for Evidently AI)
│
├── templates/
│   ├── index.html                # Loan application form (11 fields)
│   └── result.html               # Prediction result page
│
├── static/
│   └── style.css                 # Web UI styling
│
├── artifact/                     # Generated during training
│   ├── data_ingestion/           # Raw + split CSV files
│   ├── data_validation/          # validation report.yaml
│   ├── data_transformation/      # train/test .npy + preprocessing.pkl
│   ├── model_trainer/            # trained model.pkl
│   └── model_registry/           # Champion model (model.pkl + preprocessing.pkl)
│
├── logs/                         # Rotating log files (5MB each, 3 backups)
│
└── notebook/
    ├── train.csv                 # Raw training data
    ├── exp-notebook.ipynb        # EDA & feature engineering notebook
    └── postgresql_conn.ipynb     # PostgreSQL ingestion notebook
```

---

## Pipeline Architecture

```
PostgreSQL (loan_data table)
        │
        ▼
┌─────────────────────────┐
│  1. DATA INGESTION      │
│  - Fetch all rows       │
│  - Save to feature store│
│  - Stratified 75/25     │
│    train/test split      │
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  2. DATA VALIDATION     │
│  - Column count check   │
│  - Column names check   │
│  - Data types check     │
│  - Domain values check  │
│  - Missing value report │
│  - Logged to MLflow     │
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  3. DATA TRANSFORMATION │
│  - Engineered features: │
│    TotalIncome, EMI,    │
│    BalanceIncome,       │
│    Loan_Income_Ratio    │
│  - Impute missing vals  │
│  - Scale numerical feats│
│  - Ordinal encode cat   │
│  - Save preprocessing   │
│    pipeline (pickle)     │
│  - Logged to MLflow     │
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  4. MODEL TRAINER       │
│  - Apply SMOTE (balance)│
│  - Train model (config) │
│  - Optimal threshold    │
│    via precision-recall │
│  - Metrics: accuracy,   │
│    precision, recall,   │
│    F1, ROC-AUC          │
│  - Confusion matrix plot│
│  - Save model (pickle)  │
│  - Logged to MLflow     │
│    (params, metrics,    │
│     model, plots)       │
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  5. MODEL EVALUATION    │
│  - Compare vs champion  │
│  - Accept if Δ ≥ 2%     │
│  - Register in MLflow   │
│    Model Registry       │
│    (Staging alias)      │
│  - Logged to MLflow     │
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  6. MODEL PUSHER        │
│  - Copy to local        │
│    model_registry/      │
│  - (or cloud: GCS/Azure)│
└─────────────────────────┘
          │
          ▼
┌─────────────────────────┐
│  PREDICTION PIPELINE    │
│  - Load model.pkl       │
│  - Load preprocessing   │
│  - Transform input      │
│  - Predict              │
│  - Return "Y" or "N"    │
└─────────────────────────┘
```

### Training Pipeline Flow (`demo.py`)

```python
from src.pipline.training_pipeline import TrainingPipeline

pipeline = TrainingPipeline()
artifact = pipeline.run_pipeline()
```

### Prediction Flow (`app.py`)

```
POST /predict
  → CustomData (form fields → DataFrame)
  → PredictionPipeline.predict(data)
    → Load model.pkl + preprocessing.pkl from artifact/
    → preprocessor.transform(df)
    → model.predict(transformed)
    → inverse map: 1→"Y", 0→"N"
  → Render result.html
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- PostgreSQL (local or remote) with a `loan_data` table seeded
- pip

### Step 1: Clone & Virtual Environment

```bash
git clone <repo-url>
cd MLOps

# Create virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# Create virtual environment (Linux/macOS)
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt

# Install the local package (editable mode)
pip install -e .
```

### Step 3: Configure PostgreSQL

Create a `.env` file in the project root:

```env
PG_HOST=localhost
PG_PORT=5432
PG_DATABASE=mlops
PG_USER=postgres
PG_PASSWORD=your_password
```

Ensure your PostgreSQL instance has a `loan_data` table with 13 columns matching `config/schema.yaml`. Seed it with the Kaggle loan prediction dataset.

### Step 4: Run Training Pipeline

```bash
python demo.py
```

### Step 5: Start Flask Web App

```bash
python app.py
# Open http://127.0.0.1:5000
```

### Step 6: View MLflow Dashboard

```bash
mlflow ui --port 5001 --host 127.0.0.1
# Open http://127.0.0.1:5001
```

---

## Configuration

### Data Schema (`config/schema.yaml`)

Defines all columns, data types, numerical/categorical splits, which columns to drop, and valid domain values:

```yaml
columns:
  - Loan_ID: str
  - Gender: str
  - Married: str
  - ...
target_column: Loan_Status
numerical_columns: [ApplicantIncome, CoapplicantIncome, LoanAmount, ...]
categorical_columns: [Gender, Married, Dependents, ...]
drop_columns: [Loan_ID]
domain_values:
  Gender: [Male, Female]
  Married: [Yes, No]
  ...
```

### Model Selection (`config/model.yaml`)

Switch models by changing the `model_name` field:

```yaml
model_name: random_forest    # or logistic_regression, decision_tree,
                              # gradient_boosting, knn, svm
random_state: 101
n_estimators: 300
max_depth: 8
criterion: entropy
...
```

### Constants (`src/constants/__init__.py`)

Centralizes all paths, database keys, MLflow config, and hyperparameters:

```python
ARTIFACT_DIR = "artifact"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO = 0.25
MODEL_TRAINER_EXPECTED_SCORE = 0.6
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE = 0.02
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "LoanPrediction"
MLFLOW_REGISTERED_MODEL_NAME = "LoanPredictionModel"
```

---

## How to Run

### Train the Model

```bash
python demo.py
```

Output example:
```
Model trained with accuracy: 0.7597
Model saved at: artifact\model_trainer\trained_model\model.pkl
```

### Make Predictions (Web UI)

```bash
python app.py
# Open http://127.0.0.1:5000
```

Fill the form fields and submit to see the loan decision.

### Make Predictions (cURL)

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -d "gender=Male" \
  -d "married=Yes" \
  -d "dependents=0" \
  -d "education=Graduate" \
  -d "self_employed=No" \
  -d "applicant_income=5000" \
  -d "coapplicant_income=0" \
  -d "loan_amount=150" \
  -d "loan_amount_term=360" \
  -d "credit_history=1.0" \
  -d "property_area=Urban"
```

### View MLflow UI

```bash
mlflow ui --port 5001
# http://127.0.0.1:5001
```

---

## MLflow Experiment Tracking

Every pipeline run is automatically logged to MLflow with:

### Logged Parameters
- Model type and hyperparameters (from `config/model.yaml`)
- Numerical and categorical feature lists
- Engineered feature names
- Number of features after transform
- Train/test data file paths

### Logged Metrics
- `train_accuracy`, `test_accuracy`
- `test_precision`, `test_recall`
- `test_f1`, `test_roc_auc`
- `best_threshold` (optimal decision threshold)
- `validation_status`
- `smote_samples_majority`, `smote_samples_minority`
- `existing_model_accuracy`, `changed_accuracy`
- `model_accepted`

### Logged Artifacts
| Artifact | Path |
|----------|------|
| Validation report | `artifacts/validation/report.yaml` |
| Preprocessing pipeline | `artifacts/preprocessor/preprocessing.pkl` |
| Classification report | `artifacts/metrics/classification_report.txt` |
| Confusion matrix plot | `artifacts/confusion_matrix.png` |
| Trained model (sklearn format) | `artifacts/model/` |
| Raw model.pkl | `artifacts/model/model.pkl` |

---

## MLflow Model Registry

When a trained model is accepted (first run or improvement ≥ 2%), it is automatically:

1. **Registered** as `LoanPredictionModel` in the MLflow Model Registry
2. **Aliased** as `staging`

```python
result = mlflow.register_model(model_uri, "LoanPredictionModel")
client.set_registered_model_alias(name="LoanPredictionModel", version=result.version, alias="staging")
```

### Query Registered Models

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient("sqlite:///mlflow.db")
versions = client.search_model_versions('name="LoanPredictionModel"')
for v in versions:
    print(f"{v.name} v{v.version} - alias: {v.aliases}")
```

### Promote from Staging to Production (via UI or API)

```python
client.set_registered_model_alias(
    name="LoanPredictionModel",
    version=1,
    alias="production",
)
```

---

## Web Interface

### Home Page (`/`)
Loan application form with 11 input fields:
- Gender, Married, Dependents, Education, Self_Employed
- ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term
- Credit_History, Property_Area

### Result Page (`/predict` POST)
Displays:
- The prediction ("Approved" / "Rejected")
- All input details for verification

---

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/` | Renders the loan application form |
| `POST` | `/predict` | Accepts form data, returns prediction |
| — | `404` | Renders form with "Page not found." error |
| — | `500` | Renders form with internal server error message |

---

## Models Supported

Change model via `config/model.yaml`:

| Key (`model_name`) | Class | Key Hyperparameters |
|---------------------|-------|---------------------|
| `random_forest` (default) | `RandomForestClassifier` | n_estimators, max_depth, min_samples_split, criterion |
| `logistic_regression` | `LogisticRegression` | C, max_iter |
| `decision_tree` | `DecisionTreeClassifier` | max_depth, min_samples_split, criterion |
| `gradient_boosting` | `GradientBoostingClassifier` | n_estimators, learning_rate, max_depth |
| `knn` | `KNeighborsClassifier` | n_neighbors, weights |
| `svm` | `SVC` | C, kernel |

All tree/logistic models use `class_weight="balanced"` to handle class imbalance.

---

## Feature Engineering

Four derived features are created during Data Transformation:

| Feature | Formula |
|---------|---------|
| `TotalIncome` | `ApplicantIncome + CoapplicantIncome` |
| `EMI` | `LoanAmount / (Loan_Amount_Term / 12)` |
| `BalanceIncome` | `TotalIncome - (EMI * 1000)` |
| `Loan_Income_Ratio` | `LoanAmount / (TotalIncome + 1)` |

### Preprocessing Pipeline

```
ColumnTransformer(
    ("numerical_pipeline", Pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
    ), numerical_columns),
    ("categorical_pipeline", Pipeline(
        SimpleImputer(strategy="most_frequent"),
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    ), categorical_columns),
)
```

---

## Model Evaluation

### SMOTE (Synthetic Minority Oversampling)

Balances the target classes before training:
```
Before SMOTE:  {0: 144 (N), 1: 316 (Y)}  ← imbalanced
After SMOTE:   {0: 316 (N), 1: 316 (Y)}   ← balanced
```

### Optimal Threshold Tuning

Uses precision-recall curve on training data to find the decision threshold that maximizes F1-score, rather than using the default 0.5.

### Champion vs. Challenger

The evaluation component compares the new model's test accuracy against the existing model in the local registry. The new model is accepted only if:

```
new_accuracy - existing_accuracy >= 0.02  (2% improvement)
```

---

## Logging & Error Handling

### Logger (`src/logger/`)

- **Rotating file handler**: logs written to `logs/MM_DD_YYYY_HH_MM_SS.log`, max 5MB per file, 3 backups
- **Console handler**: colored output (green=INFO, yellow=WARNING, red=ERROR, cyan=DEBUG)
- Format: `[ timestamp ] name - LEVEL - message`

### Custom Exception (`src/exception/`)

`MyException` captures:
- File name where the error occurred
- Line number
- Error message

Automatically logs the error on creation.

---

## Deployment Notes

### Production Server

Instead of Flask's development server:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PG_HOST` | `localhost` | PostgreSQL host |
| `PG_PORT` | `5432` | PostgreSQL port |
| `PG_DATABASE` | `postgres` | Database name |
| `PG_USER` | `postgres` | Database user |
| `PG_PASSWORD` | `postgres` | Database password |
| `PORT` | `5000` | Flask app port |

### Model Registry Backup

The local filesystem registry at `artifact/model_registry/` stores the champion model independently from MLflow — provides fallback for the prediction pipeline.

---

## Future Improvements

- [ ] **CI/CD Pipeline**: GitHub Actions for automated training on data changes
- [ ] **Model Monitoring**: Re-integrate Evidently AI for data/model drift detection
- [ ] **Docker Containerization**: Dockerfile + docker-compose for reproducible deployment
- [ ] **Unit Tests**: pytest suite for all components
- [ ] **Hyperparameter Tuning**: Automated grid search / Optuna integration
- [ ] **Feature Store**: Centralized feature store for production serving
- [ ] **A/B Testing**: Route traffic between model versions
