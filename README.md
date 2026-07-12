# Integrating Continuous Integration (CI) into IRIS ML Pipeline

## Overview

This assignment implements a Continuous Integration (CI) workflow for an Iris classification machine learning pipeline using **GitHub Actions**.

In previous stages of the project, the ML pipeline was enhanced with:

* **Vertex AI** for cloud-based model training
* **DVC** for dataset and model versioning
* **Feast Feature Store** for feature management and training-serving consistency

This assignment focuses on adding CI automation to ensure that every code change is automatically validated before being merged.

The CI pipeline performs:

* Data validation testing
* Model evaluation testing
* Automated test execution using GitHub Actions
* Continuous Machine Learning (CML) reporting on Pull Requests

---

# Objectives

The main objectives of this assignment are:

* Build automated validation tests for ML datasets
* Validate trained model performance automatically
* Configure GitHub Actions for CI execution
* Integrate DVC-tracked ML artifacts into the workflow
* Generate automated test reports using CML
* Ensure quality checks before merging changes

---

# Project Architecture

```
Developer Push / Pull Request
              |
              |
              v
       GitHub Actions CI
              |
              |
       Install Dependencies
              |
              |
              v
          Run Pytest
              |
       ----------------
       |              |
       v              v
 Data Validation   Model Evaluation
       |              |
       ----------------
              |
              v
          CML Report
              |
              v
       Pull Request Review
             
```

---

# Repository Structure

```
21F3000371_MLOPS_WEEKLY_ASSIGNMENT
│
├── data/
│   ├── iris.csv
│   └── iris.csv.dvc
│
├── models/
│   ├── model.pkl
│   └── model.pkl.dvc
│
├── src/
│   ├── train.py
│   └── inference.py
│
├── tests/
│   ├── test_data.py
│   └── test_model.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── requirements.txt
│
└── README.md
```

---

# Dataset

The project uses the Iris dataset containing four numerical features:

| Feature      | Description     |
| ------------ | --------------- |
| sepal_length | Length of sepal |
| sepal_width  | Width of sepal  |
| petal_length | Length of petal |
| petal_width  | Width of petal  |

Target variable:

```
species
```

The dataset is version controlled using **DVC**.

Example:

```
data/iris.csv
data/iris.csv.dvc
```

The `.dvc` file stores metadata required to reproduce the dataset version.

---

# Model

The pipeline uses a:

```
Random Forest Classifier
```

The trained model is stored as:

```
models/model.pkl
```

The model artifact is tracked using DVC:

```
models/model.pkl.dvc
```

---

# Testing Framework

Testing is implemented using:

```
pytest
```

The test suite contains two categories:

---

## 1. Data Validation Tests

File:

```
tests/test_data.py
```

The tests verify:

* Dataset availability
* Correct column schema
* Missing value checks
* Feature value ranges

Example validations:

```python
assert expected_columns == actual_columns
assert df.isnull().sum().sum() == 0
```

---

## 2. Model Evaluation Tests

File:

```
tests/test_model.py
```

The model test performs:

1. Loading the trained model
2. Loading evaluation data
3. Generating predictions
4. Calculating accuracy
5. Checking minimum performance threshold

Example:

```python
assert accuracy >= 0.8
```

---

# Continuous Integration Workflow

The CI workflow is configured using:

```
.github/workflows/ci.yml
```

The workflow runs automatically on:

* Push events
* Pull Requests

Workflow steps:

## 1. Checkout Repository

Uses:

```
actions/checkout
```

to fetch the repository.

## 2. Setup Python Environment

Creates a Python environment:

```
Python 3.12
```

## 3. Install Dependencies

Installs required packages:

```
pip install -r requirements.txt
```

## 4. Execute Tests

Runs:

```
pytest
```

The workflow fails automatically if any test fails.

---

# CML Integration

Continuous Machine Learning (CML) is integrated into the workflow to generate automated reports.

After testing:

* Test results are collected
* A report is generated
* Results are posted as a Pull Request comment

Example report:

```
IRIS Model CI Report

Test Results

5 passed
```

This allows reviewers to verify ML pipeline quality before merging.

---

# Running Locally

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Tests

```bash
pytest
```

Expected output:

```
5 passed
```

---

# Git Workflow

Development was performed on:

```
week_4
```

branch.

After successful CI validation:

Pull Request workflow:

```
Create PR
    |
    v
GitHub Actions CI
    |
    v
CML Report
    |
    v
Merge
```

---

# Technologies Used

| Technology            | Purpose                   |
| --------------------- | ------------------------- |
| Python                | Programming language      |
| Scikit-learn          | Machine learning model    |
| Pytest                | Automated testing         |
| GitHub Actions        | Continuous Integration    |
| DVC                   | Data and model versioning |
| CML                   | ML reporting              |
| Google Cloud Platform | Cloud ML environment      |

---

# CI Execution Result

Final pipeline status:

```
✓ Data validation tests passed
✓ Model evaluation test passed
✓ GitHub Actions workflow passed
✓ CML report generated successfully
```

---

# Conclusion

This assignment demonstrates a complete CI workflow for a machine learning pipeline.

By integrating GitHub Actions and CML, every code change is automatically validated through data checks and model performance tests before merging.

This ensures better reliability, reproducibility, and quality control in ML development workflows.
