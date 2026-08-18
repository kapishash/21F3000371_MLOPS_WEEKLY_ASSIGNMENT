import vertexai
from vertexai.tuning import sft

PROJECT_ID = "mlops-500009"
LOCATION = "us-central1"

TRAIN_DATASET = (
    "gs://21f3000371-iris-data/week10/iris_v1_gemini.jsonl"
)

vertexai.init(
    project=PROJECT_ID,
    location=LOCATION,
)

print("Starting Gemini V1 supervised fine-tuning...")

job = sft.train(
    source_model="gemini-2.5-flash",
    train_dataset=TRAIN_DATASET,
    tuned_model_display_name="iris-gemini-v1-raw",
    tuning_mode="PEFT_ADAPTER",
    epochs=3,
    learning_rate_multiplier=1.0,
)

print("Tuning job submitted successfully.")
print("Job:", job)
print("Resource name:", job.resource_name)
