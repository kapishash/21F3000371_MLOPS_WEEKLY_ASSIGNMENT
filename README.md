# IRIS MLOps Pipeline – Week 7

## Overview

This project implements a complete MLOps pipeline for the IRIS Classification model on Google Cloud Platform.

The pipeline includes:

- Model training
- Model versioning
- Artifact Storage
- Docker containerization
- Kubernetes deployment (GKE)
- CI/CD using GitHub Actions
- Horizontal Pod Autoscaling
- Stress Testing
- Cloud Monitoring
- Cloud Logging

---

# Architecture

GitHub
↓
GitHub Actions (CI/CD)
↓
Docker Build
↓
Artifact Registry
↓
Google Kubernetes Engine
↓
IRIS Prediction API
↓
Horizontal Pod Autoscaler
↓
Cloud Monitoring & Cloud Logging

---

# Technologies Used

- Python
- FastAPI
- Docker
- Kubernetes (GKE)
- Google Artifact Registry
- GitHub Actions
- Horizontal Pod Autoscaler
- Google Cloud Monitoring
- Google Cloud Logging
- wrk

---

# Deployment

Build Docker Image

```bash
docker build -t iris-api .
```

Run locally

```bash
docker run -p 8000:8000 iris-api
```

Deploy

```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

---

# Horizontal Pod Autoscaler

Created using

```bash
kubectl autoscale deployment iris-api \
--cpu-percent=50 \
--min=1 \
--max=3
```

Verification

```bash
kubectl get hpa
```

---

# Stress Testing

wrk was used to generate concurrent HTTP traffic.

## POST Script

```lua
wrk.method = "POST"

wrk.body = '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'

wrk.headers["Content-Type"] = "application/json"
```

---

## Test 1

1000 concurrent users

```bash
wrk -t4 -c1000 -d30s -s post.lua http://<EXTERNAL_IP>/predict
```

Observed

- HPA scaled from 1 Pod to 3 Pods.
- CPU utilization exceeded the target threshold.
- Additional Pods were created automatically.
- Service handled increased load better than the constrained setup.

---

## Test 2

Single Replica Bottleneck

HPA configuration

```
minReplicas = 1
maxReplicas = 1
```

Stress Test

```bash
wrk -t4 -c2000 -d30s -s post.lua http://<EXTERNAL_IP>/predict
```

Observed

- No additional Pods were created.
- CPU became saturated.
- High socket read/write errors occurred.
- Throughput dropped significantly.
- The single Pod became the performance bottleneck.

---

# Monitoring

During stress testing the following GCP services were used.

- Cloud Monitoring
- Cloud Logging

Observed Metrics

- CPU Utilization
- Memory Usage
- Pod Scaling
- Request Logs
- Container Logs

---

# CI/CD

GitHub Actions automatically performs

- Checkout Repository
- Authenticate to GCP
- Build Docker Image
- Push to Artifact Registry
- Deploy to GKE
- Wait for rollout
- Install wrk
- Execute stress testing

---

# Bottleneck Analysis

Scenario 1

- maxReplicas = 3
- 1000 concurrent users

Result

- Pods scaled automatically.
- Better throughput.
- Lower request failures.

Scenario 2

- maxReplicas = 1
- 2000 concurrent users

Result

- No autoscaling.
- CPU saturation.
- High socket errors.
- Throughput degraded significantly.

This demonstrates how Horizontal Pod Autoscaling improves service reliability under heavy load.

---

# Repository

https://github.com/kapishash/21F3000371_MLOPS_WEEKLY_ASSIGNMENT