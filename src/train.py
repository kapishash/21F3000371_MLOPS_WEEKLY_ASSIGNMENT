from google.cloud import storage
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
import os
from datetime import datetime

DATA_BUCKET = "21f3000371-iris-data"
ARTIFACT_BUCKET = "21f3000371-iris-artifacts"

client = storage.Client()

# Download dataset
data_bucket = client.bucket(DATA_BUCKET)
blob = data_bucket.blob("data_v2.csv") # data_v2.csv for 2nd version
blob.download_to_filename("data_v2.csv")  # data_v2.csv for 2nd version

# Read dataset
df = pd.read_csv("data_v2.csv")

# Split data
train_df, eval_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["species"]
)

# Training
X_train = train_df.drop("species", axis=1)
y_train = train_df["species"]

X_eval = eval_df.drop("species", axis=1)
y_eval = eval_df["species"]

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_eval)
accuracy = accuracy_score(y_eval, preds)

# Timestamp folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

os.makedirs("outputs", exist_ok=True)

model_path = "outputs/model.pkl"
eval_path = "outputs/eval.csv"
metrics_path = "outputs/metrics.txt"

joblib.dump(model, model_path)
eval_df.to_csv(eval_path, index=False)

with open(metrics_path, "w") as f:
    f.write(f"Accuracy: {accuracy}")

# Upload artifacts
artifact_bucket = client.bucket(ARTIFACT_BUCKET)

artifact_bucket.blob(
    f"{timestamp}/model.pkl"
).upload_from_filename(model_path)

artifact_bucket.blob(
    f"{timestamp}/eval.csv"
).upload_from_filename(eval_path)

artifact_bucket.blob(
    f"{timestamp}/metrics.txt"
).upload_from_filename(metrics_path)

print("Training completed")
print("Accuracy:", accuracy)
print("Artifacts stored in:", timestamp)