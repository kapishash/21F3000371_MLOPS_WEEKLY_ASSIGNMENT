# IRIS MLOps Pipeline — Week 9

## Explainability, Fairness, and Drift Monitoring

This week extends the IRIS machine learning pipeline with responsible ML practices covering:

* Explainability using SHAP
* Fairness evaluation using Fairlearn
* Data drift detection using statistical tests
* ML model governance using a Model Card

The work is implemented on the `week_9` branch of the IRIS MLOps repository.

---

## Week 9 Objectives

The main objectives of this assignment are:

1. Introduce a sensitive `location` attribute into the IRIS dataset.
2. Evaluate fairness across location groups using Fairlearn.
3. Generate SHAP explanations for the IRIS classifier.
4. Analyze the factors influencing Virginica predictions.
5. Simulate production data drift.
6. Detect feature drift using the Kolmogorov-Smirnov statistical test.
7. Document the model using a Model Card.

---

# Project Structure

```text
21F3000371_MLOPS_WEEKLY_ASSIGNMENT
│
├── data/
│   ├── iris.csv
│   ├── iris.csv.dvc
│   ├── iris1.csv
│   └── week9/
│       └── iris_with_location.csv
│
├── src/
│   ├── train.py
│   ├── inference.py
│   ├── poison_data.py
│   ├── train_mlsecops.py
│   ├── add_location.py
│   ├── fairness.py
│   ├── explainability.py
│   └── drift.py
│
├── outputs/
│   └── drift/
│       ├── iris_production.csv
│       ├── sepal_length_drift.png
│       ├── sepal_width_drift.png
│       ├── petal_length_drift.png
│       └── petal_width_drift.png
│
├── MODEL_CARD.md
├── requirements.txt
├── Dockerfile
├── deployment.yaml
├── service.yaml
├── hpa.yaml
└── README.md
```

---

# Task 1 — Introduce Location Attribute

A new `location` column was added to the IRIS dataset.

The value is randomly assigned as either:

```text
0
1
```

The location attribute is used only for fairness analysis and is not included as a model training feature.

## Dataset

The resulting dataset contains:

```text
190 samples
```

The location distribution was:

```text
location
0    93
1    97
```

The resulting file is:

```text
data/week9/iris_with_location.csv
```

## Generate the Dataset

Run:

```bash
python src/add_location.py
```

Expected output:

```text
Dataset created successfully
Total samples: 190

Location distribution:
location
0    93
1    97
```

---

# Task 2 — Fairness Evaluation

Fairness was evaluated using:

```text
Fairlearn
```

The `MetricFrame` API was used to calculate performance separately for location groups `0` and `1`.

The following metrics were evaluated:

* Accuracy
* Precision
* Recall

The location attribute was not used during model training.

## Run Fairness Analysis

```bash
python src/fairness.py
```

## Results

Overall metrics:

```text
Accuracy : 1.0
Precision: 1.0
Recall   : 1.0
```

Fairlearn MetricFrame:

```text
          accuracy  precision  recall
location
0              1.0        1.0     1.0
1              1.0        1.0     1.0
```

Metric differences:

```text
accuracy     0.0
precision    0.0
recall       0.0
```

Therefore, no measurable performance difference was observed between the two randomly assigned location groups in this experiment.

Because the location attribute was randomly generated, this experiment demonstrates the fairness auditing process rather than proving real-world demographic fairness.

---

# Task 3 — Model Explainability Using SHAP

Model explainability was implemented using:

```text
SHAP
```

SHAP provides feature-level explanations for model predictions.

The analysis generates explanations for the IRIS classification model and focuses on the three classes:

```text
setosa
versicolor
virginica
```

The four model features are:

```text
sepal_length
sepal_width
petal_length
petal_width
```

## SHAP Interpretation

For the Virginica class:

* Positive SHAP values push the prediction toward Virginica.
* Negative SHAP values push the prediction away from Virginica.
* Red points represent high feature values.
* Blue points represent low feature values.
* A point on the right side of the plot indicates a positive contribution toward Virginica.
* A point on the left side indicates a negative contribution.

A cluster of red points on the right indicates that high values of that feature provide evidence supporting the Virginica prediction.

## Generate SHAP Plots

Run:

```bash
python src/explainability.py
```

The generated plots can be used during the screencast to explain the model's decision-making process.

---

# Task 4 — Data Drift Detection

Production data was simulated by shifting the feature distributions of the IRIS dataset.

The original training distribution was compared against the simulated production distribution.

The:

```text
Kolmogorov-Smirnov (KS) test
```

was used to detect statistical differences between the distributions.

## Run Drift Analysis

```bash
python src/drift.py
```

## Drift Results

The analysis produced the following results:

| Feature      | KS Statistic |  P-value | Result         |
| ------------ | -----------: | -------: | -------------- |
| sepal_length |       0.2211 | 0.000176 | DRIFT DETECTED |
| sepal_width  |       0.3474 | 0.000000 | DRIFT DETECTED |
| petal_length |       0.3263 | 0.000000 | DRIFT DETECTED |
| petal_width  |       0.2684 | 0.000002 | DRIFT DETECTED |

All four features showed statistically significant distribution changes.

The drift plots are generated in:

```text
outputs/drift/
```

Generated files include:

```text
iris_production.csv
sepal_length_drift.png
sepal_width_drift.png
petal_length_drift.png
petal_width_drift.png
```

---

# Data Drift vs Concept Drift

## Data Drift

Data drift occurs when the distribution of input features changes.

For example:

```text
Training data → Normal petal length distribution

Production data → Shifted petal length distribution
```

Statistical tests such as the KS test can be used to detect this change.

## Concept Drift

Concept drift occurs when the relationship between the features and target changes.

For example, the same feature values may begin corresponding to different classes in production.

Concept drift generally requires monitoring model predictions against ground-truth labels.

---

# Task 5 — Model Card and Governance

A Model Card was created to document the IRIS classifier.

The file is:

```text
MODEL_CARD.md
```

The Model Card contains:

* Model purpose
* Intended use
* Training data
* Model information
* Overall performance
* Fairness results
* Explainability information
* Drift results
* Limitations
* Fairness considerations
* Monitoring recommendations
* Governance considerations

The Model Card improves transparency and accountability for the ML system.

---

# Week 9 Workflow

```text
                 IRIS Dataset
                      |
                      v
            Add Location Attribute
                      |
                      v
             Train ML Classifier
                      |
          ┌───────────┼───────────┐
          |           |           |
          v           v           v
      Fairness     SHAP       Drift
      Analysis   Explainability Detection
          |           |           |
          v           v           v
       Fairlearn    SHAP        KS Test
          |           |           |
          └───────────┼───────────┘
                      |
                      v
                 Model Card
                      |
                      v
                ML Governance
```

---

# Key Findings

## Fairness

The Fairlearn analysis showed identical performance for location groups `0` and `1`:

```text
Accuracy  = 1.00
Precision = 1.00
Recall    = 1.00
```

The metric difference between the groups was:

```text
0.00
```

for all three metrics.

---

## Explainability

SHAP was used to understand how the four flower measurements contribute to classification.

For Virginica predictions:

* Positive SHAP values support Virginica.
* Negative SHAP values move predictions away from Virginica.
* High feature values are represented by red points.
* Low feature values are represented by blue points.

---

## Drift

All four monitored features showed statistically significant drift:

```text
sepal_length  → Drift detected
sepal_width   → Drift detected
petal_length  → Drift detected
petal_width   → Drift detected
```

This indicates that the simulated production distribution differs from the original training distribution.

---

# Limitations

This experiment has several limitations:

1. The IRIS dataset is small.
2. The location attribute is randomly generated.
3. Location does not represent an actual demographic group.
4. The fairness experiment therefore demonstrates the auditing methodology rather than real-world demographic fairness.
5. The production drift was artificially simulated.
6. Detecting feature drift does not automatically mean that model accuracy has decreased.
7. Concept drift requires ground-truth labels and performance monitoring.

---

# Production Recommendations

A production ML system should continuously monitor:

* Feature distributions
* Model performance
* Fairness metrics
* Prediction distributions
* Data quality
* Training-data provenance
* Model versions

When significant drift is detected, the team should investigate the cause before automatically retraining the model.

Fairness audits should also be performed periodically across relevant groups.

SHAP or other explainability methods can be used to monitor whether the model's decision patterns change over time.

---

# Week 9 Commands

## Create Location Dataset

```bash
python src/add_location.py
```

## Run Fairness Analysis

```bash
python src/fairness.py
```

## Generate SHAP Explanations

```bash
python src/explainability.py
```

## Detect Data Drift

```bash
python src/drift.py
```

## Check Generated Drift Files

```bash
ls -lh outputs/drift/
```

## Check Git Status

```bash
git status
```

---
