# Integrating MLSecOps into the IRIS Pipeline (Week 8)

## Overview

This assignment extends the IRIS machine learning pipeline with **MLSecOps (Machine Learning Security Operations)** practices.

The objective is to understand security threats across the ML lifecycle and demonstrate a practical **data poisoning attack** against the IRIS classification dataset.

Four datasets were evaluated:

* Clean dataset — 0% poisoning
* 5% poisoned dataset
* 10% poisoned dataset
* 50% poisoned dataset

Each dataset was used to train an IRIS classification model, and the experiments were tracked using **MLflow**.

The assignment demonstrates how compromised training data can affect model performance and why data validation, provenance tracking, anomaly detection, and quality gates are important in production ML systems.

---

# Objectives

The main objectives of this assignment are:

* Understand major ML security threat vectors
* Simulate data poisoning attacks
* Generate poisoned IRIS datasets at different severity levels
* Train models using clean and poisoned datasets
* Track experiments using MLflow
* Compare model performance across poisoning levels
* Analyze the impact of data quality degradation
* Identify mitigation strategies for production ML systems
* Understand the relationship between data quantity and data quality

---

# MLSecOps

MLSecOps applies security principles throughout the machine learning lifecycle.

The major attack surfaces considered in this assignment are:

```text
Data Ingestion
      |
      v
Data Poisoning
      |
      v
Model Training
      |
      v
Model Artifacts
      |
      v
Model Extraction
      |
      v
Inference API
      |
      v
Adversarial Examples
      |
      v
LLM / Text Interfaces
      |
      v
Prompt Injection
```

---

# ML Security Threat Vectors

## 1. Data Poisoning

### Target Stage

Data ingestion and model training.

### Description

Data poisoning occurs when an attacker intentionally modifies training data to degrade model performance or influence model behavior.

In this assignment, poisoning is simulated by replacing selected IRIS samples with:

* Random feature values
* Random class labels

Three poisoning levels were evaluated:

```text
5%
10%
50%
```

### Example Scenario

An attacker gains access to a training data pipeline and modifies a portion of the dataset before model training.

The resulting model learns from incorrect patterns and may produce incorrect predictions.

### Mitigation

Possible defenses include:

* Data validation
* Statistical profiling
* Anomaly detection
* Data provenance tracking
* Schema validation
* Dataset versioning
* Quality gates before training

---

# 2. Adversarial Examples

### Target Stage

Model inference.

### Description

Adversarial examples are inputs deliberately modified to cause a trained model to make an incorrect prediction.

Unlike data poisoning, the attacker does not necessarily modify the training dataset.

Instead, the attack happens during inference.

### Example

A small change to the numerical features supplied to the IRIS API could potentially cause a model to predict the wrong class.

### Mitigation

Possible approaches include:

* Input validation
* Range checking
* Outlier detection
* Adversarial robustness testing
* Rate limiting
* Monitoring unusual inference patterns

---

# 3. Model Extraction

### Target Stage

Model serving and inference.

### Description

Model extraction occurs when an attacker repeatedly queries a deployed model and attempts to reconstruct its behavior.

For example, an attacker could send a large number of carefully selected requests to a prediction API and use the responses to build a substitute model.

### Mitigation

Possible defenses include:

* Rate limiting
* Authentication
* Query monitoring
* Access controls
* Limiting prediction information
* Logging inference requests

---

# 4. Prompt Injection

### Target Stage

LLM or natural-language inference systems.

### Description

Prompt injection occurs when malicious instructions are included in user input to manipulate an AI system into ignoring its intended behavior.

Although the current IRIS pipeline is not an LLM application, prompt injection is an important MLSecOps threat for systems that accept natural-language input.

### Mitigation

Possible approaches include:

* Input filtering
* Prompt boundary enforcement
* Instruction hierarchy
* Output validation
* Access control
* Monitoring suspicious inputs

---

# Dataset

The project uses the IRIS classification dataset.

The dataset contains four numerical features:

| Feature      | Description  |
| ------------ | ------------ |
| sepal_length | Sepal length |
| sepal_width  | Sepal width  |
| petal_length | Petal length |
| petal_width  | Petal width  |

Target variable:

```text
species
```

Classes:

```text
setosa
versicolor
virginica
```

The clean dataset contains:

```text
190 samples
```

excluding the header.

---

# Data Poisoning Simulation

The poisoning attack was implemented using:

```text
src/poison_data.py
```

For each poisoning level, a percentage of the dataset samples is selected.

For every selected sample:

1. The four numerical features are replaced with randomly generated values.
2. The original class label is replaced with a randomly selected class.
3. The modified dataset is saved as a separate CSV file.

The original clean dataset is preserved as the baseline.

---

# Poisoning Levels

Four dataset variants were created:

| Dataset      | Poisoning Level | Modified Samples |
| ------------ | --------------: | ---------------: |
| Clean        |              0% |                0 |
| Poisoned 5%  |              5% |                9 |
| Poisoned 10% |             10% |               19 |
| Poisoned 50% |             50% |               95 |

The poisoning script produced the following files:

```text
data/mlsecops/
├── iris_clean.csv
├── iris_poisoned_5.csv
├── iris_poisoned_10.csv
└── iris_poisoned_50.csv
```

Each dataset contains 190 samples.

Including the header, each CSV contains:

```text
191 lines
```

---

# Poisoning Script

The poisoning process is implemented in:

```text
src/poison_data.py
```

It generates the three poisoned datasets automatically.

Run:

```bash
python src/poison_data.py
```

Expected output:

```text
Clean dataset: 190 samples
Classes: ['setosa', 'versicolor', 'virginica']

5% poisoning: 9 samples modified
10% poisoning: 19 samples modified
50% poisoning: 95 samples modified
```

---

# Model Training

The MLsecOps training workflow is implemented in:

```text
src/train_mlsecops.py
```

The training process:

1. Loads a dataset variant.
2. Splits the data into training and evaluation sets.
3. Trains a classification model.
4. Generates predictions.
5. Calculates evaluation metrics.
6. Logs the experiment to MLflow.
7. Records the poisoning level as an MLflow parameter.

The four experiments correspond to:

```text
0% poisoning
5% poisoning
10% poisoning
50% poisoning
```

---

# MLflow Experiment Tracking

MLflow was used to track the poisoning experiments.

Experiment name:

```text
iris_mlsecops_poisoning
```

The MLflow backend was configured using SQLite:

```text
sqlite:///mlflow.db
```

The MLflow UI was started using:

```bash
mlflow ui \
  --backend-store-uri sqlite:///mlflow.db \
  --host 0.0.0.0 \
  --port 5000
```

The experiment comparison UI was used to compare model performance across all poisoning levels.

---

# Metrics Tracked

For every experiment, the following metrics were tracked:

* Accuracy
* Precision
* Recall
* F1 Score

The poisoning level was logged as an experiment parameter.

This allows the effect of increasing data corruption to be compared systematically.

---

# MLflow Experiment Comparison

The MLflow experiment contains four runs:

```text
Clean
5% Poisoning
10% Poisoning
50% Poisoning
```

The MLflow comparison interface allows the metrics of all four runs to be viewed together.

The comparison helps determine:

* When model degradation begins
* Which metrics change first
* How severe the degradation becomes
* Whether the model continues learning meaningful patterns under heavy corruption

---

# Analysis of Poisoning Impact

The clean dataset provides the baseline model performance.

At low poisoning levels, the model may still learn useful patterns because most of the training data remains clean.

As poisoning increases:

```text
Clean data ratio
       |
       v
100% → 95% → 90% → 50%
       |
       v
Increasing data corruption
```

The 50% poisoning scenario represents a severe attack because half of the training samples have been replaced with randomly generated features and labels.

At this level, the training data contains a large amount of incorrect information, making it significantly more difficult for the model to learn the genuine relationships between IRIS features and species.

The MLflow comparison provides empirical evidence for evaluating this degradation.

---

# Why Data Quality Matters

Increasing the amount of training data does not automatically solve a data-quality problem.

Consider a dataset containing:

```text
1000 samples
```

If 10% of the data is poisoned:

```text
900 clean
100 poisoned
```

Adding more data is useful only when the additional data is trustworthy.

For example, adding another 1000 poisoned samples would make the situation worse rather than better.

Therefore:

```text
More Data ≠ Better Model
```

when the additional data is unreliable.

The focus should first be on ensuring data quality and provenance.

---

# Data Quantity vs Data Quality

The relationship can be summarized as:

```text
Clean Data
    |
    v
Reliable Training Signal
    |
    v
Better Model Learning
```

Whereas:

```text
Poisoned Data
    |
    v
Incorrect Training Signal
    |
    v
Model Degradation
```

As the poisoned-data ratio increases, the amount of trustworthy information available to the model decreases.

For a fixed total dataset size:

| Poisoning | Clean Ratio |
| --------: | ----------: |
|        0% |        100% |
|        5% |         95% |
|       10% |         90% |
|       50% |         50% |

Therefore, when data quality is compromised, the required total dataset size may need to increase to maintain a sufficient amount of clean training data.

However, simply collecting more data is not an effective mitigation if the new data can also be poisoned.

---

# Task 5: Mitigation Strategies

A production ML pipeline should introduce security controls before corrupted data reaches model training.

## 1. Statistical Validation

Monitor:

* Mean
* Median
* Standard deviation
* Feature distributions
* Class distributions

Unexpected distribution changes can indicate data corruption.

---

## 2. Anomaly Detection

Outlier detection can identify samples that significantly differ from the expected feature distribution.

For example:

```text
Normal IRIS feature range
          |
          v
Expected samples
          |
          X
Unexpected extreme values
```

Suspicious samples can be flagged for investigation.

---

## 3. Data Provenance Tracking

Every dataset should have information about:

* Source
* Creation time
* Version
* Processing history
* Responsible pipeline
* Hash/checksum

DVC can help maintain dataset versions and reproducibility.

---

## 4. Schema Enforcement

The pipeline should verify that incoming data has:

* Expected columns
* Correct data types
* Valid feature ranges
* Valid target classes
* No unexpected missing values

For example:

```text
sepal_length → numerical
sepal_width  → numerical
petal_length → numerical
petal_width  → numerical
species      → valid class
```

---

## 5. Data Quality Gates

Training should not begin automatically if validation fails.

A secure pipeline can follow:

```text
Data Ingestion
      |
      v
Schema Validation
      |
      v
Statistical Validation
      |
      v
Anomaly Detection
      |
      v
Quality Gate
      |
   Pass / Fail
      |
      v
Model Training
```

This prevents suspicious datasets from entering the training stage.

---

# Production MLSecOps Architecture

A secure ML pipeline can be structured as:

```text
             Data Source
                 |
                 v
          Data Validation
                 |
        -------------------
        |                 |
      Valid             Invalid
        |                 |
        v                 v
   Data Versioning     Reject / Alert
        |
        v
   Model Training
        |
        v
   Model Validation
        |
        v
    Model Registry
        |
        v
    Deployment
        |
        v
   Inference API
        |
        v
 Monitoring & Logging
```

Security controls should exist at every stage.

---

# Repository Structure

```text
21F3000371_MLOPS_WEEKLY_ASSIGNMENT/
│
├── data/
│   ├── iris.csv
│   ├── iris.csv.dvc
│   └── mlsecops/
│       ├── iris_clean.csv
│       ├── iris_poisoned_5.csv
│       ├── iris_poisoned_10.csv
│       └── iris_poisoned_50.csv
│
├── src/
│   ├── train.py
│   ├── inference.py
│   ├── poison_data.py
│   └── train_mlsecops.py
│
├── tests/
│   ├── test_data.py
│   └── test_model.py
│
├── models/
│   └── model.pkl
│
├── artifacts/
│   ├── model.pkl
│   ├── eval.csv
│   └── metrics.txt
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
│
├── deployment.yaml
├── service.yaml
├── hpa.yaml
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

---



# Reproducing the Experiment

## 1. Generate Poisoned Datasets

```bash
python src/poison_data.py
```

---

## 2. Run MLsecOps Training

```bash
python src/train_mlsecops.py
```

The training script evaluates the clean and poisoned datasets and records the experiments in MLflow.

---

## 3. Start MLflow

```bash
mlflow ui \
  --backend-store-uri sqlite:///mlflow.db \
  --host 0.0.0.0 \
  --port 5000
```

Open the MLflow interface and select:

```text
iris_mlsecops_poisoning
```

Compare the four runs:

```text
0%
5%
10%
50%
```

---


# Conclusion

This assignment demonstrates the importance of security throughout the machine learning lifecycle.

A data poisoning attack was simulated on the IRIS dataset at:

```text
5%
10%
50%
```

corruption levels, with the clean dataset used as the baseline.

MLflow was used to track and compare the resulting model experiments using accuracy, precision, recall, and F1 score.

The experiment demonstrates that model reliability depends not only on the amount of training data but also on the quality and trustworthiness of that data.

A production ML pipeline should therefore combine:

```text
Data Validation
+
Data Provenance
+
Anomaly Detection
+
Schema Enforcement
+
Quality Gates
+
Model Validation
+
Monitoring
```

to reduce the risk of ML-specific security threats.

MLSecOps extends traditional DevSecOps principles to the complete ML lifecycle, protecting the system from data-level attacks through to model serving and inference.
