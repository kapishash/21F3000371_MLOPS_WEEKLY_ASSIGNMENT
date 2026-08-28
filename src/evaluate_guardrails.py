import csv
import os
import sys
from datetime import datetime, timezone

import pandas as pd

# Make src imports work when executed from repository root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from input_guardrail import guarded_predict
from output_guardrail import guard_output


# ============================================================
# CONFIGURATION
# ============================================================

INJECTION_FILE = "data/week11/injection_redteam_results.csv"
LEAKAGE_FILE = "data/week11/leakage_redteam_results.csv"

# Week 10 evaluation file used previously.
WEEK10_FILE = "data/week10/week10_evaluation_results.csv"

OUTPUT_FILE = "data/week11/guardrail_effectiveness_results.csv"
SUMMARY_FILE = "data/week11/guardrail_effectiveness_summary.csv"

VALID_SPECIES = {
    "setosa",
    "versicolor",
    "virginica",
}

FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]


# ============================================================
# INPUT FORMATTER
# ============================================================

def iris_prompt(row):
    """
    Convert a Week 10 row into the input format expected
    by the Task 3 input guardrail.
    """

    return (
        f"sepal_length: {row['sepal_length']}, "
        f"sepal_width: {row['sepal_width']}, "
        f"petal_length: {row['petal_length']}, "
        f"petal_width: {row['petal_width']}"
    )


# ============================================================
# GUARDED PIPELINE
# ============================================================

def run_guarded_pipeline(raw_input, model_version):
    """
    Full Task 5 pipeline:

        raw input
            ↓
        input guardrail
            ↓
        Vertex AI model
            ↓
        output guardrail
            ↓
        final response
    """

    input_result = guarded_predict(
        raw_input,
        model_version=model_version,
    )

    if input_result.get("blocked"):
        return {
            "input_blocked": True,
            "input_reason": input_result.get("reason"),
            "model_response": "",
            "output_filtered": False,
            "output_reason": None,
            "final_response": "",
        }

    model_response = input_result.get(
        "prediction",
        "",
    )

    output_result = guard_output(
        model_response
    )

    return {
        "input_blocked": False,
        "input_reason": None,
        "model_response": model_response,
        "output_filtered": output_result["filtered"],
        "output_reason": output_result["reason"],
        "final_response": output_result["response"],
    }


# ============================================================
# RUN ADVERSARIAL TESTS
# ============================================================

def run_adversarial_tests():

    injection_df = pd.read_csv(INJECTION_FILE)
    leakage_df = pd.read_csv(LEAKAGE_FILE)

    results = []

    # --------------------------------------------------------
    # Task 1 — injection attacks
    # --------------------------------------------------------

    for _, row in injection_df.iterrows():

        raw_input = row["input_prompt"]
        model_version = row["model_version"]

        result = run_guarded_pipeline(
            raw_input,
            model_version,
        )

        results.append(
            {
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
                "category": "injection",
                "attack_pattern": row["attack_pattern"],
                "model_version": model_version,
                "input": raw_input,
                "input_blocked": result["input_blocked"],
                "input_reason": result["input_reason"],
                "model_response": result["model_response"],
                "output_filtered": result["output_filtered"],
                "output_reason": result["output_reason"],
                "final_response": result["final_response"],
            }
        )

    # --------------------------------------------------------
    # Task 2 — leakage attacks
    # --------------------------------------------------------

    for _, row in leakage_df.iterrows():

        raw_input = row["input_prompt"]
        model_version = row["model_version"]

        result = run_guarded_pipeline(
            raw_input,
            model_version,
        )

        results.append(
            {
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
                "category": "leakage",
                "attack_pattern": row["attack_pattern"],
                "model_version": model_version,
                "input": raw_input,
                "input_blocked": result["input_blocked"],
                "input_reason": result["input_reason"],
                "model_response": result["model_response"],
                "output_filtered": result["output_filtered"],
                "output_reason": result["output_reason"],
                "final_response": result["final_response"],
            }
        )

    return results


# ============================================================
# WEEK 10 CLEAN TEST
# ============================================================

def run_clean_tests():

    df = pd.read_csv(WEEK10_FILE)

    results = []

    for _, row in df.iterrows():

        raw_input = iris_prompt(row)

        # Evaluate both model versions.
        for model_version in ["v1", "v2"]:

            result = run_guarded_pipeline(
                raw_input,
                model_version,
            )

            final_response = (
                result["final_response"]
                .strip()
                .lower()
            )

            actual = str(
                row["actual"]
            ).strip().lower()

            correct = (
                not result["input_blocked"]
                and not result["output_filtered"]
                and final_response == actual
            )

            results.append(
                {
                    "timestamp": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "category": "clean",
                    "attack_pattern": "",
                    "model_version": model_version,
                    "input": raw_input,
                    "actual": actual,
                    "input_blocked": result["input_blocked"],
                    "input_reason": result["input_reason"],
                    "model_response": result["model_response"],
                    "output_filtered": result["output_filtered"],
                    "output_reason": result["output_reason"],
                    "final_response": result["final_response"],
                    "correct": correct,
                }
            )

    return results


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        fieldnames = sorted(
            {
                key
                for row in results
                for key in row.keys()
            }
        )

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)


# ============================================================
# WEEK 10 BASELINE
# ============================================================

def calculate_week10_baseline():

    df = pd.read_csv(WEEK10_FILE)

    # V1 and V2 predictions from the original Week 10 run.
    v1_accuracy = (
        df["v1_prediction"].astype(str).str.lower()
        == df["actual"].astype(str).str.lower()
    ).mean()

    v2_accuracy = (
        df["v2_prediction"].astype(str).str.lower()
        == df["actual"].astype(str).str.lower()
    ).mean()

    return {
        "v1": v1_accuracy,
        "v2": v2_accuracy,
    }


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    adversarial_results,
    clean_results,
):

    baseline = calculate_week10_baseline()

    summary = []

    # --------------------------------------------------------
    # Injection block rate
    # --------------------------------------------------------

    injection = [
        r
        for r in adversarial_results
        if r["category"] == "injection"
    ]

    injection_block_rate = (
        sum(r["input_blocked"] for r in injection)
        / len(injection)
        if injection
        else 0
    )

    # --------------------------------------------------------
    # Leakage block rate
    #
    # A leakage attempt is considered blocked when:
    #   input was blocked OR
    #   output was filtered.
    # --------------------------------------------------------

    leakage = [
        r
        for r in adversarial_results
        if r["category"] == "leakage"
    ]

    leakage_block_rate = (
        sum(
            r["input_blocked"]
            or r["output_filtered"]
            for r in leakage
        )
        / len(leakage)
        if leakage
        else 0
    )

    # --------------------------------------------------------
    # Clean false-positive rate
    # --------------------------------------------------------

    false_positive_rates = {}

    for version in ["v1", "v2"]:

        version_results = [
            r
            for r in clean_results
            if r["model_version"] == version
        ]

        false_positives = sum(
            r["input_blocked"]
            for r in version_results
        )

        false_positive_rates[version] = (
            false_positives / len(version_results)
            if version_results
            else 0
        )

    # --------------------------------------------------------
    # Guarded accuracy
    # --------------------------------------------------------

    guarded_accuracy = {}

    for version in ["v1", "v2"]:

        version_results = [
            r
            for r in clean_results
            if r["model_version"] == version
        ]

        guarded_accuracy[version] = (
            sum(
                r["correct"]
                for r in version_results
            )
            / len(version_results)
            if version_results
            else 0
        )

    # --------------------------------------------------------
    # Accuracy delta
    # --------------------------------------------------------

    accuracy_delta = {
        version:
            guarded_accuracy[version]
            - baseline[version]
        for version in ["v1", "v2"]
    }

    # --------------------------------------------------------
    # Summary rows
    # --------------------------------------------------------

    summary.append(
        {
            "metric": "injection_block_rate",
            "v1": injection_block_rate,
            "v2": injection_block_rate,
        }
    )

    summary.append(
        {
            "metric": "leakage_block_rate",
            "v1": leakage_block_rate,
            "v2": leakage_block_rate,
        }
    )

    summary.append(
        {
            "metric": "false_positive_rate",
            "v1": false_positive_rates["v1"],
            "v2": false_positive_rates["v2"],
        }
    )

    summary.append(
        {
            "metric": "week10_accuracy",
            "v1": baseline["v1"],
            "v2": baseline["v2"],
        }
    )

    summary.append(
        {
            "metric": "guarded_accuracy",
            "v1": guarded_accuracy["v1"],
            "v2": guarded_accuracy["v2"],
        }
    )

    summary.append(
        {
            "metric": "accuracy_delta",
            "v1": accuracy_delta["v1"],
            "v2": accuracy_delta["v2"],
        }
    )

    with open(
        SUMMARY_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "metric",
                "v1",
                "v2",
            ],
        )

        writer.writeheader()
        writer.writerows(summary)

    return summary


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("WEEK 11 — TASK 5 GUARDRAIL EFFECTIVENESS")
    print("=" * 70)

    print("\nRunning adversarial tests...")

    adversarial_results = run_adversarial_tests()

    print(
        f"Completed {len(adversarial_results)} "
        "adversarial evaluations."
    )

    print("\nRunning Week 10 clean-input tests...")

    clean_results = run_clean_tests()

    print(
        f"Completed {len(clean_results)} "
        "clean-input evaluations."
    )

    all_results = (
        adversarial_results
        + clean_results
    )

    save_results(all_results)

    summary = calculate_metrics(
        adversarial_results,
        clean_results,
    )

    print("\n" + "=" * 70)
    print("TASK 5 SUMMARY")
    print("=" * 70)

    for row in summary:

        print(
            f"{row['metric']:<25}"
            f"V1: {row['v1']:.2%}    "
            f"V2: {row['v2']:.2%}"
        )

    print("\nDetailed results:")
    print(OUTPUT_FILE)

    print("\nSummary:")
    print(SUMMARY_FILE)

    print("=" * 70)
