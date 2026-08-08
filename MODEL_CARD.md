# Model Card: IRIS Classification Model

## 1. Model Overview

The IRIS classifier is a machine learning model designed to classify IRIS flowers into one of three species:

* Setosa
* Versicolor
* Virginica

The model uses a Random Forest Classifier trained on four numerical flower measurements.

## 2. Intended Use

The model is intended for:

* Demonstrating multiclass classification
* Learning ML explainability and fairness analysis
* Studying data drift in an ML production pipeline
* Educational MLOps experimentation

The model is not intended for high-stakes real-world decision making.

## 3. Training Data

The original IRIS dataset contains 190 samples in this project.

The model uses the following four features:

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

A `location` attribute was added for fairness evaluation. It was randomly assigned as either `0` or `1`.

The `location` attribute was not used as a model feature.

## 4. Model

Model type:

```text
Random Forest Classifier
```

The classifier uses the four original IRIS measurements to predict the flower species.

## 5. Overall Performance

The fairness evaluation produced the following overall metrics:

| Metric    | Score |
| --------- | ----: |
| Accuracy  |  1.00 |
| Precision |  1.00 |
| Recall    |  1.00 |

These results represent the evaluation performed in the Week 9 experiment.

## 6. Fairness Evaluation

Fairness was evaluated using Fairlearn's `MetricFrame`.

The sensitive attribute was:

```text
location
```

The attribute was randomly assigned and was excluded from model training.

### Performance by Location

| Location | Accuracy | Precision | Recall |
| -------- | -------: | --------: | -----: |
| 0        |     1.00 |      1.00 |   1.00 |
| 1        |     1.00 |      1.00 |   1.00 |

Metric differences between the two groups were:

| Metric    | Difference |
| --------- | ---------: |
| Accuracy  |       0.00 |
| Precision |       0.00 |
| Recall    |       0.00 |

The experiment therefore showed no measurable performance disparity between the two randomly assigned location groups.

## 7. Explainability

SHAP was used to explain the model's predictions.

SHAP summary plots were generated for the three IRIS classes:

* Setosa
* Versicolor
* Virginica

The SHAP analysis helps identify how each feature contributes toward or away from a particular prediction.

For the Virginica class:

* Positive SHAP values push the prediction toward Virginica.
* Negative SHAP values push the prediction away from Virginica.
* Red points represent high feature values.
* Blue points represent low feature values.

The SHAP plots provide a post-hoc explanation of the Random Forest predictions.

## 8. Data Drift Evaluation

A simulated production dataset was created by shifting the feature distributions from the original dataset.

The Kolmogorov-Smirnov test was used to compare the original and production distributions.

| Feature      | KS Statistic | Result         |
| ------------ | -----------: | -------------- |
| sepal_length |       0.2211 | Drift detected |
| sepal_width  |       0.3474 | Drift detected |
| petal_length |       0.3263 | Drift detected |
| petal_width  |       0.2684 | Drift detected |

All four features produced statistically significant drift with p-values below 0.05.

This indicates that the simulated production data differs significantly from the training distribution.

## 9. Limitations

The model and evaluation have several limitations:

* The dataset is small.
* The `location` attribute is randomly generated and does not represent a real demographic attribute.
* The fairness experiment therefore demonstrates the auditing process rather than proving real-world fairness.
* The simulated production drift is artificially created.
* Drift detection alone does not prove that model accuracy has decreased.
* The model was evaluated on the specific dataset used in this assignment and may not generalize to different flower populations or measurement conditions.

## 10. Fairness Considerations

The `location` attribute was intentionally excluded from model training and was used only for fairness evaluation.

Because the attribute was randomly assigned, equal performance across groups was expected.

In a real-world system, sensitive attributes should be carefully governed, and fairness should be evaluated using meaningful demographic or protected-group information where legally and ethically appropriate.

## 11. Monitoring Recommendations

For production deployment, the following monitoring practices are recommended:

* Monitor feature distributions continuously.
* Detect statistically significant data drift.
* Monitor model performance when ground-truth labels become available.
* Track fairness metrics across relevant groups.
* Monitor SHAP feature importance for unexpected changes.
* Maintain model version and training-data provenance.
* Retrain or investigate the model when significant drift or performance degradation is detected.

## 12. Governance

The model should be accompanied by documentation describing:

* Intended use
* Training data
* Model architecture
* Performance metrics
* Fairness results
* Explainability results
* Known limitations
* Drift monitoring results

This documentation supports transparency and accountability throughout the ML lifecycle.

## 13. Conclusion

The IRIS pipeline demonstrates responsible ML practices by combining classification, fairness auditing, explainability, and drift monitoring.

The Week 9 experiments showed:

* Equal measured performance across the two randomly assigned location groups.
* SHAP-based explanations of model predictions.
* Significant distribution drift across all four input features in the simulated production dataset.

These analyses demonstrate why model accuracy alone is not sufficient for trustworthy machine learning deployment.
