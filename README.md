# Week 10 — LLMOps: Fine-Tuning and Evaluation of Gemini Models

## Overview

This assignment extends the IRIS machine learning pipeline into an **LLMOps workflow** using Google Cloud Vertex AI / Agent Platform.

The objective is to fine-tune a Gemini model for IRIS species classification and compare two different ways of representing the same training data:

* **V1 — Raw Feature Representation**
* **V2 — Natural Language Description**

Both versions use supervised fine-tuning with LoRA. The models are evaluated on the **same held-out test set** using:

* Accuracy
* Per-class precision
* Per-class recall
* Format compliance rate

The assignment demonstrates how data representation can affect the behavior of a fine-tuned LLM, including an LLM-specific issue that traditional classifiers do not normally have: **output-format compliance**.

---

# Objectives

The main objectives of Week 10 are:

* Understand LLMOps and supervised fine-tuning
* Prepare structured IRIS data for Gemini fine-tuning
* Compare raw feature prompts with natural-language prompts
* Create a held-out train/test split
* Convert datasets into Gemini supervised-tuning JSONL format
* Fine-tune Gemini models using LoRA
* Deploy the tuned models on Vertex AI
* Run predictions against an independent test set
* Calculate accuracy, precision, recall, and format compliance
* Compare V1 and V2
* Analyze the effect of data representation on LLM behavior
* Demonstrate reproducible model evaluation

---

# LLMOps Workflow

The Week 10 workflow can be summarized as:

```text
IRIS Dataset
     |
     v
Train/Test Split
     |
     +----------------------+
     |                      |
     v                      v
V1 Raw Features       V2 Natural Language
     |                      |
     v                      v
Gemini JSONL           Gemini JSONL
     |                      |
     v                      v
LoRA Fine-Tuning       LoRA Fine-Tuning
     |                      |
     v                      v
Vertex AI Models       Vertex AI Models
     |                      |
     +----------+-----------+
                |
                v
       Held-Out Test Set
                |
                v
        Model Evaluation
                |
                v
 Accuracy / Precision / Recall
       / Format Compliance
                |
                v
        V1 vs V2 Comparison
```

---

# Dataset

The project uses the IRIS classification dataset.

The dataset contains four numerical features:

| Feature | Description |
| --- | --- |
| `sepal_length` | Sepal length in cm |
| `sepal_width` | Sepal width in cm |
| `petal_length` | Petal length in cm |
| `petal_width` | Petal width in cm |

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

The working dataset contains:

```text
190 samples
```

excluding the CSV header.

---

# Week 10 Train/Test Split

To avoid evaluating the tuned models on their training examples, a held-out test set was created.

The split contains:

```text
Total samples: 190
Training samples: 152
Test samples: 38
```

The class distributions are:

## Training Set

| Species | Samples |
| --- | ---: |
| setosa | 50 |
| versicolor | 52 |
| virginica | 50 |
| **Total** | **152** |

## Test Set

| Species | Samples |
| --- | ---: |
| setosa | 12 |
| versicolor | 13 |
| virginica | 13 |
| **Total** | **38** |

The test set is kept separate from the fine-tuning data and is used only for final evaluation.

The split was created using:

```text
src/create_week10_split.py
```

Run:

```bash
python src/create_week10_split.py
```

Output:

```text
data/week10/train.csv
data/week10/test.csv
```

---

# Data Representation

Two representations of the same IRIS information were created.

The purpose is to investigate whether the way information is presented to an LLM affects its fine-tuned behavior.

---

# V1 — Raw Feature Representation

V1 represents the input directly as structured feature-value pairs.

Example:

```text
sepal_length: 6.3, sepal_width: 3.3, petal_length: 4.7, petal_width: 1.6
```

Expected output:

```text
versicolor
```

This representation is concise and closely resembles a traditional tabular ML input.

The V1 dataset was generated using:

```text
src/prepare_llm_data.py
src/convert_v1_to_gemini.py
```

The final Gemini training format is:

```json
{
  "contents": [
    {
      "role": "user",
      "parts": [
        {
          "text": "sepal_length: 6.3, sepal_width: 3.3, petal_length: 4.7, petal_width: 1.6"
        }
      ]
    },
    {
      "role": "model",
      "parts": [
        {
          "text": "versicolor"
        }
      ]
    }
  ]
}
```

---

# V2 — Natural Language Description

V2 converts the same numerical features into a natural-language description.

Example:

```text
A flower specimen has a sepal length of 6.3 cm,
sepal width of 3.3 cm, petal length of 4.7 cm,
and petal width of 1.6 cm. Identify the iris species.
```

Expected output:

```text
This is Iris versicolor.
```

This representation is more natural for a language model because the input resembles normal language rather than a raw feature vector.

The V2 dataset was generated using:

```text
src/prepare_llm_data.py
src/convert_v2_to_gemini.py
```

The final Gemini training format is:

```json
{
  "contents": [
    {
      "role": "user",
      "parts": [
        {
          "text": "A flower specimen has a sepal length of 6.3 cm, sepal width of 3.3 cm, petal length of 4.7 cm, and petal width of 1.6 cm. Identify the iris species."
        }
      ]
    },
    {
      "role": "model",
      "parts": [
        {
          "text": "This is Iris versicolor."
        }
      ]
    }
  ]
}
```

---

# Why Two Representations?

The underlying data is identical.

Only the input/output representation changes.

```text
Same IRIS data
      |
      +------------------------+
      |                        |
      v                        v
V1: Raw Features       V2: Natural Language
      |                        |
      v                        v
Structured Input        Linguistic Input
```

This makes the comparison useful because differences in model behavior can be associated with the representation rather than with a completely different dataset.

---

# Gemini Fine-Tuning

The base model used for the assignment is:

```text
gemini-2.5-flash
```

Fine-tuning method:

```text
Supervised Fine-Tuning
```

Tuning mode:

```text
LoRA
```

LoRA is a parameter-efficient fine-tuning technique that trains a smaller set of adapter parameters rather than updating the complete base model.

This reduces the computational cost of fine-tuning while allowing the model to specialize for the IRIS classification task.

---

# V1 Fine-Tuning

The V1 tuning workflow is implemented in:

```text
src/tune_v1.py
src/tune_v1_final.py
```

The final V1 model was trained using the V1 training dataset:

```text
gs://21f3000371-iris-data/week10/iris_v1_train_gemini.jsonl
```

The final tuning job was successfully submitted and completed.

The tuned model was deployed as:

```text
iris-gemini-v1-final
```

---

# V2 Fine-Tuning

The V2 tuning workflow is implemented in:

```text
src/tune_v2.py
src/tune_v2_final.py
```

The final V2 model was trained using:

```text
gs://21f3000371-iris-data/week10/iris_v2_train_gemini.jsonl
```

The tuned model was successfully trained and deployed as:

```text
iris-gemini-v2-final
```

---

# Vertex AI Deployment

The tuned models were deployed to Vertex AI / Agent Platform for online prediction.

The final deployments included:

```text
iris-gemini-v1-final
iris-gemini-v2-final
```

The deployment region was:

```text
us-central1
```

The models were then accessed by the evaluation script to generate predictions for the held-out test set.

---

# Test Script

A small sanity test was also implemented using:

```text
src/test_tuned_model.py
```

Example test cases include:

```text
5.1, 3.5, 1.4, 0.2 → setosa
6.7, 3.1, 4.7, 1.5 → versicolor
6.3, 3.3, 6.0, 2.5 → virginica
```

The test script verified that the tuned model could correctly classify representative IRIS samples.

---

# Final Evaluation

The complete held-out evaluation was performed using:

```text
src/evaluate_week10.py
```

Run:

```bash
python src/evaluate_week10.py
```

The script evaluates both tuned models against exactly the same 38 test samples.

For every test sample, the script records:

* Input features
* Expected species
* V1 raw output
* V1 prediction
* V1 format compliance
* V2 raw output
* V2 prediction
* V2 format compliance

Detailed results are saved to:

```text
data/week10/week10_evaluation_results.csv
```

---

# Evaluation Metrics

## 1. Accuracy

Accuracy measures the percentage of test samples classified correctly.

```text
Accuracy = Correct Predictions / Total Predictions
```

---

## 2. Precision

Precision measures how many predictions for a particular class were actually correct.

```text
Precision = True Positives / (True Positives + False Positives)
```

Precision is calculated separately for:

```text
setosa
versicolor
virginica
```

---

## 3. Recall

Recall measures how many actual samples of a class were correctly identified.

```text
Recall = True Positives / (True Positives + False Negatives)
```

Recall is also calculated separately for each species.

---

# 4. Format Compliance

Format compliance is an LLM-specific evaluation metric required for this assignment.

A response is considered compliant only if the model outputs exactly one valid species name:

```text
setosa
versicolor
virginica
```

The following are considered non-compliant:

```text
Setosa
Virginica
This is setosa.
The answer is versicolor.
{"species": "virginica"}
```

Even when the semantic prediction is correct, these outputs fail the required format.

This is important because a traditional classifier normally returns a predefined class label, whereas an LLM generates text.

Therefore:

```text
Correct Classification ≠ Format Compliance
```

Both dimensions need to be evaluated.

---

# Final Task 4 Results

The held-out test set contained:

```text
38 samples
```

Test distribution:

```text
setosa      : 12
versicolor  : 13
virginica   : 13
```

The final evaluation produced:

| Metric | V1 Raw | V2 Description |
| --- | ---: | ---: |
| Accuracy | 100.00% | 100.00% |
| Format Compliance | 100.00% | 97.37% |

## Per-Class Precision

| Class | V1 | V2 |
| --- | ---: | ---: |
| setosa | 100.00% | 100.00% |
| versicolor | 100.00% | 100.00% |
| virginica | 100.00% | 100.00% |

## Per-Class Recall

| Class | V1 | V2 |
| --- | ---: | ---: |
| setosa | 100.00% | 100.00% |
| versicolor | 100.00% | 100.00% |
| virginica | 100.00% | 100.00% |

---

# V1 vs V2 Comparison

The final comparison is:

```text
                     V1 Raw       V2 Description

Accuracy             100.00%         100.00%
Format Compliance    100.00%          97.37%
```

Both models achieved perfect classification accuracy on the held-out test set.

However, V1 achieved better format compliance.

Therefore, based on the required evaluation metrics:

```text
Winner: V1 Raw Feature Representation
```

The difference was not in classification accuracy, precision, or recall.

The difference was in the generated output format.

---

# Observed V2 Format Issue

One V2 prediction illustrates the LLM-specific failure mode.

Expected:

```text
virginica
```

V2 produced:

```text
Virginica
```

The semantic prediction is correct, but the output uses an uppercase first letter.

According to the assignment definition:

```text
Virginica
```

is non-compliant because the expected output must match the valid species name exactly.

Therefore:

```text
Prediction: Correct
Format: Incorrect
```

This reduced V2's format compliance to:

```text
37 / 38 = 97.37%
```

V1 produced compliant outputs for all 38 test samples:

```text
38 / 38 = 100%
```

---

# Why Did the Data Representation Make a Difference?

The two models received the same underlying numerical information, but the language patterns they were trained on were different.

## V1

V1 was trained using:

```text
sepal_length: ...
sepal_width: ...
petal_length: ...
petal_width: ...
```

and the target was simply:

```text
setosa
versicolor
virginica
```

Therefore, the model learned a very direct mapping:

```text
Feature Values → Exact Class Label
```

The desired output format was extremely simple and consistent.

---

## V2

V2 was trained using natural-language descriptions:

```text
A flower specimen has a sepal length of ...
Identify the iris species.
```

with outputs such as:

```text
This is Iris versicolor.
```

The model therefore had more natural-language patterns to reproduce.

This can make the model more likely to produce linguistic variations such as:

```text
Virginica
This is Iris virginica.
The species is virginica.
```

Even if the semantic prediction is correct, these variations can violate a strict output contract.

---

# Key Insight

The important observation is:

```text
V1 and V2 learned the classification task equally well,
but V1 followed the exact output format more reliably.
```

This demonstrates that LLM evaluation should not rely only on semantic correctness.

For production LLM systems, output formatting can be just as important as the underlying prediction.

For example, a downstream application may require:

```json
{
  "species": "virginica"
}
```

If the model instead returns:

```text
The flower appears to be Virginica.
```

the answer may be semantically correct but unusable by a strict API.

---

# LLM-Specific Failure Mode

Traditional classifiers generally return one of the predefined classes:

```text
setosa
versicolor
virginica
```

An LLM, however, generates free-form text.

Therefore, it can introduce:

* Capitalization differences
* Extra explanations
* JSON
* Punctuation
* Different wording
* Unexpected tokens

This is why format compliance was included as a separate metric.

The Week 10 experiment demonstrates:

```text
Semantic Accuracy
        +
Output Format Reliability
```

should both be considered when evaluating LLM applications.

---

# Reproducibility

The complete workflow can be reproduced using the following scripts.

## 1. Prepare LLM Data

```bash
python src/prepare_llm_data.py
```

This creates:

```text
data/week10/iris_v1_raw.jsonl
data/week10/iris_v2_description.jsonl
```

---

## 2. Create Train/Test Split

```bash
python src/create_week10_split.py
```

Creates:

```text
data/week10/train.csv
data/week10/test.csv
```

---

## 3. Prepare Gemini Training Data

Use:

```bash
python src/prepare_week10_train_data.py
```

The resulting datasets include:

```text
data/week10/iris_v1_train_gemini.jsonl
data/week10/iris_v2_train_gemini.jsonl
```

---

## 4. Convert V1 and V2 Data

The conversion scripts are:

```bash
python src/convert_v1_to_gemini.py
python src/convert_v2_to_gemini.py
```

---

## 5. Fine-Tune V1

```bash
python src/tune_v1_final.py
```

---

## 6. Fine-Tune V2

```bash
python src/tune_v2_final.py
```

The tuning jobs are executed on Vertex AI / Agent Platform.

---

## 7. Test a Tuned Model

```bash
python src/test_tuned_model.py
```

---

## 8. Run Final Evaluation

```bash
python src/evaluate_week10.py
```

The evaluation compares both models on the same held-out test set.

---

# Google Cloud Storage Artifacts

The Week 10 Gemini datasets were uploaded to:

```text
gs://21f3000371-iris-data/week10/
```

The main training artifacts include:

```text
iris_v1_gemini.jsonl
iris_v1_raw.jsonl
iris_v1_train_gemini.jsonl
iris_v2_description.jsonl
iris_v2_gemini.jsonl
iris_v2_train_gemini.jsonl
```

---

# Repository Structure

The Week 10 additions include:

```text
21F3000371_MLOPS_WEEKLY_ASSIGNMENT/
│
├── data/
│   └── week10/
│       ├── iris_v1_gemini.jsonl
│       ├── iris_v1_raw.jsonl
│       ├── iris_v1_train_gemini.jsonl
│       ├── iris_v2_description.jsonl
│       ├── iris_v2_gemini.jsonl
│       ├── iris_v2_train_gemini.jsonl
│       ├── train.csv
│       ├── test.csv
│       └── week10_evaluation_results.csv
│
├── src/
│   ├── prepare_llm_data.py
│   ├── create_week10_split.py
│   ├── prepare_week10_train_data.py
│   ├── convert_v1_to_gemini.py
│   ├── convert_v2_to_gemini.py
│   ├── tune_v1.py
│   ├── tune_v1_final.py
│   ├── tune_v2.py
│   ├── tune_v2_final.py
│   ├── test_tuned_model.py
│   └── evaluate_week10.py
│
└── README.md
```

---

# Important Experimental Considerations

The evaluation was performed on a small IRIS dataset with only 38 held-out test examples.

Therefore, the results should be interpreted in the context of this assignment rather than as evidence that one representation is universally better for all LLM applications.

The most meaningful observation is that:

```text
Both representations achieved perfect classification accuracy,
but V1 provided stricter output-format compliance.
```

The single V2 format failure demonstrates why output validation is important in production LLM systems.

---

# LLMOps Architecture

The complete Week 10 architecture can be summarized as:

```text
                 IRIS Dataset
                      |
                      v
             Train/Test Split
                      |
          +-----------+-----------+
          |                       |
          v                       v
    V1 Raw Features       V2 Description
          |                       |
          v                       v
     Gemini JSONL            Gemini JSONL
          |                       |
          v                       v
     Vertex AI SFT           Vertex AI SFT
          |                       |
          v                       v
       LoRA Model              LoRA Model
          |                       |
          +-----------+-----------+
                      |
                      v
               Vertex AI Endpoint
                      |
                      v
              Held-Out Test Set
                      |
                      v
              Prediction Engine
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
       Accuracy   Precision    Recall
                      |
                      v
              Format Compliance
                      |
                      v
                V1 vs V2
```
