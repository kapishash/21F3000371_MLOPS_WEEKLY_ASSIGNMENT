# MLOps Weekly Assignment – Week 2
## Integrating DVC into the IRIS ML Pipeline

**Student Name:** Kapish Ashtankar  
**IITM BS Roll Number:** 21F3000371

---

# Project Overview

This project demonstrates the integration of **Data Version Control (DVC)** into an end-to-end Machine Learning pipeline for the IRIS classification dataset on **Google Cloud Platform (GCP)**.

The project uses:

- Google Cloud Storage (GCS) for storing datasets, model artifacts, and DVC remote storage.
- Vertex AI Workbench as the development environment.
- DVC for versioning datasets and trained models.
- Git for source code version control.
- Scikit-learn for training the IRIS classifier.

The objective is to build a reproducible ML workflow where datasets and models can be versioned, reproduced, and restored using Git and DVC.

---

# Project Structure

```
21F3000371_MLOPS_WEEKLY_ASSIGNMENT/

│
├── .dvc/
│   └── config
│
├── src/
│   ├── train.py
│   └── inference.py
│
├── data/
│   ├── iris.csv.dvc
│   └── .gitignore
│
├── models/
│   ├── model.pkl.dvc
│   └── .gitignore
│
├── artifacts/
│   ├── eval.csv
│   └── metrics.txt
│
├── README.md
├── .dvcignore
└── .gitignore
```

---

# Technologies Used

- Python 3
- Google Cloud Platform (GCP)
- Vertex AI Workbench
- Google Cloud Storage (GCS)
- Git
- DVC (Data Version Control)
- Scikit-learn
- Pandas


---

# Google Cloud Storage Buckets

## Dataset Bucket

```
21f3000371-iris-data
```

Stores the IRIS datasets used for training.

---

## Artifact Bucket

```
21f3000371-iris-artifacts
```

Stores:

- Trained model
- Evaluation dataset
- Metrics
- Timestamp-based outputs

---

## DVC Remote Bucket

```
21f3000371-dvc-storage
```

Configured as the default DVC remote to store versioned datasets and model artifacts.

---

# Files Description

## train.py

Performs the following operations:

- Downloads dataset from Google Cloud Storage.
- Splits the dataset into training and evaluation sets.
- Trains a Random Forest classifier.
- Evaluates model performance.
- Saves:
  - model.pkl
  - evaluation dataset
  - metrics.txt
- Uploads generated artifacts to GCS.

---

## inference.py

Performs inference using the trained model by:

- Downloading the trained model.
- Loading the evaluation dataset.
- Predicting labels.
- Calculating model accuracy.

---

# DVC Workflow

The following DVC operations were performed.

Initialize DVC

```bash
dvc init
```

Configure GCS Remote

```bash
dvc remote add -d gcsremote gs://21f3000371-dvc-storage
```

Track Dataset

```bash
dvc add data/iris.csv
```

Track Model

```bash
dvc add models/model.pkl
```

Push Data and Models

```bash
dvc push
```

---

# Dataset Versioning

Two versions of the IRIS dataset were maintained.

### Version 1

Original IRIS dataset.

### Version 2

Updated IRIS dataset (second version provided in the assignment) used to retrain the model.

Each dataset version is tracked using DVC while Git stores only the corresponding `.dvc` pointer files.

---

# Model Versioning

Separate trained models were generated for each dataset version.

Each model version is tracked using DVC and stored in the configured Google Cloud Storage remote.

---

# Version Switching

Previous versions of datasets and models can be restored using:

```bash
git checkout <commit_hash>
dvc checkout
```

To return to the latest version:

```bash
git checkout week_2
dvc checkout
```

---

# Reproducing the Project

Clone the repository

```bash
git clone <repository-url>
```

Move into the repository

```bash
cd 21F3000371_MLOPS_WEEKLY_ASSIGNMENT
```

Pull DVC tracked files

```bash
dvc pull
```

Run training

```bash
python src/train.py
```

Run inference

```bash
python src/inference.py
```


**Kapish Ashtankar**

**IITM BS Roll Number:** 21F3000371
