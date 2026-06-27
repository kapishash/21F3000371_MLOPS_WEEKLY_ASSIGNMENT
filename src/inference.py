from google.cloud import storage
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import joblib
import os

ARTIFACT_BUCKET = "21f3000371-iris-artifacts"

client = storage.Client()
bucket = client.bucket(ARTIFACT_BUCKET)

# Get all blobs
blobs = list(client.list_blobs(ARTIFACT_BUCKET))

# Extract timestamp folders
folders = set()

for blob in blobs:
    if "/" in blob.name:
        folders.add(blob.name.split("/")[0])

# Get latest timestamp folder
latest_folder = sorted(folders)[-1]

print(f"Using artifacts from: {latest_folder}")

# Download model
bucket.blob(
    f"{latest_folder}/model.pkl"
).download_to_filename("model.pkl")

# Download evaluation data
bucket.blob(
    f"{latest_folder}/eval.csv"
).download_to_filename("eval.csv")

# Load model
model = joblib.load("model.pkl")

# Load evaluation data
df = pd.read_csv("eval.csv")

X = df.drop("species", axis=1)
y = df["species"]

# Predict
predictions = model.predict(X)

accuracy = accuracy_score(y, predictions)

print("\nInference Complete")
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y, predictions))