# IRIS MLOps Pipeline – Continuous Deployment (Week 6)

## Overview

This project demonstrates an end-to-end **Continuous Deployment (CD)** pipeline for an IRIS Flower Classification API using **Docker**, **Google Kubernetes Engine (GKE)**, **Google Artifact Registry**, and **GitHub Actions**.

The application exposes a REST API that predicts the Iris flower species based on four flower measurements. Every push to the repository automatically builds a Docker image, pushes it to Artifact Registry, and deploys the latest version to a Kubernetes cluster on Google Cloud.

---

## Project Architecture

```
GitHub Push
      │
      ▼
GitHub Actions
      │
      ├── Authenticate with Google Cloud (OIDC)
      ├── Build Docker Image
      ├── Push Image to Artifact Registry
      └── Deploy Updated Image to GKE
                    │
                    ▼
             Kubernetes Deployment
                    │
                    ▼
              LoadBalancer Service
                    │
                    ▼
              FastAPI Prediction API
```

---

# Technology Stack

* Python 3.12
* FastAPI
* Uvicorn
* Scikit-learn
* Docker
* Kubernetes (GKE)
* Google Cloud Platform
* Google Artifact Registry
* GitHub Actions
* Workload Identity Federation (OIDC)

---

# Project Structure

```
.
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
│
├── data/
├── models/
├── outputs/
├── src/
│   ├── train.py
│   └── inference.py
│
├── tests/
│
├── Dockerfile
├── deployment.yaml
├── service.yaml
├── requirements.txt
└── README.md
```

---

# Docker

Build the Docker image

```bash
docker build -t iris-api .
```

Run the container

```bash
docker run -p 8000:8000 iris-api
```

API URL

```
http://localhost:8000
```

Swagger UI

```
http://localhost:8000/docs
```

---

# Kubernetes Deployment

Deploy the application

```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

Verify deployment

```bash
kubectl get deployments
kubectl get pods
kubectl get services
```

---

# Docker Image

The application image is stored in **Google Artifact Registry**.

```
us-central1-docker.pkg.dev/mlops-500009/iris-repo/iris-api
```

---

# GitHub Actions Continuous Deployment

The CD pipeline automatically performs:

* Checkout repository
* Authenticate to Google Cloud using Workload Identity Federation (OIDC)
* Configure Docker
* Build Docker image
* Push image to Google Artifact Registry
* Retrieve GKE credentials
* Update Kubernetes Deployment
* Wait for successful rollout

The workflow executes automatically whenever code is pushed to the `week_6` or `main` branch.

---

# API Endpoints

## Health Check

```
GET /
```

Response

```json
{
  "message": "IRIS Prediction API is running."
}
```

---

## Predict Species

```
POST /predict
```

Request

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Response

```json
{
  "prediction": "setosa"
}
```

Example

```bash
curl -X POST http://<EXTERNAL-IP>/predict \
-H "Content-Type: application/json" \
-d '{
  "sepal_length":5.1,
  "sepal_width":3.5,
  "petal_length":1.4,
  "petal_width":0.2
}'
```

---

# Continuous Deployment Workflow

1. Push code to GitHub.
2. GitHub Actions starts automatically.
3. Authenticate with Google Cloud.
4. Build Docker image.
5. Push image to Artifact Registry.
6. Update the Kubernetes deployment.
7. Kubernetes performs a rolling update.
8. The latest API becomes available through the LoadBalancer.

---

# Pod vs Container

A **Docker container** is a lightweight package that contains an application along with all its dependencies.

A **Kubernetes Pod** is the smallest deployable unit in Kubernetes. A Pod can contain one or more containers that share networking and storage resources. Kubernetes schedules and manages Pods rather than individual containers, enabling scaling, self-healing, and rolling updates.

---

# Learning Outcomes

* Containerized an ML inference API using Docker.
* Automated image builds with GitHub Actions.
* Stored images in Google Artifact Registry.
* Configured secure authentication using Workload Identity Federation.
* Deployed the application on Google Kubernetes Engine.
* Exposed the API using a Kubernetes LoadBalancer.
* Implemented an end-to-end Continuous Deployment pipeline.

---

# Author

**Kapish Ashtankar**

IIT Madras BS Degree Programme

Course: MLOps Weekly Assignment – Week 6
