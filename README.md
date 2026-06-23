# Week 1 Assignment - IRIS ML Pipeline on Vertex AI

## Student Details

**IITM BS Roll Number:** 21F3000371

## Objective

The objective of this assignment is to build an end-to-end machine learning pipeline on Google Cloud Platform (GCP) using Vertex AI Workbench and Google Cloud Storage (GCS). The pipeline downloads the IRIS dataset from a GCS bucket, trains a machine learning model, stores artifacts in a separate GCS bucket, and performs inference using the stored model.

---

## Project Structure

```text
21f3000371_MLOPS_WEEKLY_ASSIGNMENT/

├── README.md
├── requirements.txt
│
├── src/
│   ├── train.py
│   └── inference.py
│
├── data/
│   ├── iris_v1.csv
│   └── iris_v2.csv
│
└── artifacts/
    ├── eval.csv
    ├── metrics.txt

```

---

## Google Cloud Resources

### Data Bucket

```text
gs://21f3000371-iris-data
```

Stores the IRIS dataset used for training and evaluation.

### Artifacts Bucket

```text
gs://21f3000371-iris-artifacts
```

Stores model artifacts generated during training. Each execution creates a separate timestamped folder containing:

```text
<timestamp>/
├── model.pkl
├── eval.csv
└── metrics.txt
```

This structure enables experiment traceability and reproducibility.

---

## Source Files

### src/train.py

This script:

* Connects to Google Cloud Storage
* Downloads the IRIS dataset from the data bucket
* Splits the dataset into training and evaluation sets
* Trains a Random Forest Classifier
* Evaluates model performance
* Creates a timestamp-based artifact folder
* Uploads generated artifacts to the artifacts bucket

Generated artifacts include:

* model.pkl
* eval.csv
* metrics.txt

---

### src/inference.py

This script:

* Connects to Google Cloud Storage
* Identifies the latest training execution folder
* Downloads the trained model and evaluation dataset
* Performs inference on the evaluation set
* Calculates and displays model accuracy
* Generates a classification report

---

## Data Folder

The data folder contains two versions of the IRIS dataset:

### iris.csv

Original version of the dataset used for training and experimentation.

### data_v2.csv

Alternative version of the dataset used to demonstrate handling multiple data versions as part of the optional assignment task.

---

## Artifacts Folder

The artifacts folder contains output files demonstrating successful execution of the pipeline.

### eval.csv

Evaluation dataset used during inference.

### metrics.txt

Stores evaluation metrics generated during training.

### training_output.txt

Console output generated after successful execution of the training pipeline.

### inference_output.txt

Console output generated after successful execution of the inference pipeline.

---

## Requirements

Install required dependencies:

```bash
pip install -r requirements.txt
```

Contents of requirements.txt:

```text
pandas
scikit-learn
joblib
google-cloud-storage
```

---

## Execution

### Run Training

```bash
python src/train.py
```

Example Output:

```text
Training completed
Accuracy: 0.90
Artifacts stored in: 20260621_114054
```

---

### Run Inference

```bash
python src/inference.py
```

Example Output:

```text
Using artifacts from: 20260621_114054

Inference Complete
Accuracy: 0.90
```

---

## Results

The Random Forest classifier was successfully trained on the IRIS dataset using Google Cloud infrastructure. Model artifacts were stored in Google Cloud Storage under timestamp-based folders, enabling multiple executions and experiment tracking.

The pipeline successfully demonstrates:

* Data retrieval from GCS
* Model training on Vertex AI Workbench
* Artifact storage in GCS
* Separate inference workflow
* Timestamp-based experiment organization
* Reproducible cloud-based ML pipeline

---

