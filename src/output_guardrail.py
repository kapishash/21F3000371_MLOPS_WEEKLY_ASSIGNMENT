import csv
import os
import re
from datetime import datetime, timezone


# ============================================================
# CONFIGURATION
# ============================================================

AUDIT_FILE = "data/week11/output_guardrail_audit.csv"

VALID_SPECIES = {
    "setosa",
    "versicolor",
    "virginica",
}

FALLBACK_MESSAGE = "Unable to provide a compliant classification."


# ============================================================
# CONTEXT LEAKAGE PATTERNS
# ============================================================

LEAKAGE_PATTERNS = [
    re.compile(r"\bsystem\s+prompt\b", re.IGNORECASE),
    re.compile(r"\bsystem\s+instructions?\b", re.IGNORECASE),
    re.compile(r"\bdeveloper\s+instructions?\b", re.IGNORECASE),
    re.compile(r"\bhidden\s+instructions?\b", re.IGNORECASE),
    re.compile(r"\binternal\s+instructions?\b", re.IGNORECASE),
    re.compile(r"\btraining\s+instructions?\b", re.IGNORECASE),
    re.compile(r"\binstructions\s+i\s+received\b", re.IGNORECASE),
    re.compile(r"\bmy\s+instructions\s+are\b", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+a\s+helpful\s+assistant\b", re.IGNORECASE),
    re.compile(r"\bi\s+was\s+instructed\s+to\b", re.IGNORECASE),
]


# ============================================================
# CONTEXT LEAKAGE DETECTION
# ============================================================

def detect_context_leakage(response):
    """
    Detect output that contains fragments resembling
    system, developer, hidden, or training instructions.

    Returns:
        matched pattern name, or None.
    """

    text = response.strip()

    for pattern in LEAKAGE_PATTERNS:
        if pattern.search(text):
            return pattern.pattern

    return None


# ============================================================
# FORMAT VALIDATION
# ============================================================

def validate_species_format(response):
    """
    Valid output must be exactly one of the allowed
    Iris species names.

    Case-sensitive validation is intentional because
    Week 10 format compliance required exact output.
    """

    return response.strip() in VALID_SPECIES


# ============================================================
# AUDIT LOGGING
# ============================================================

def log_filtered_response(raw_response, reason):
    """
    Log every filtered response for audit purposes.
    """

    os.makedirs(
        os.path.dirname(AUDIT_FILE),
        exist_ok=True,
    )

    file_exists = os.path.exists(AUDIT_FILE)

    with open(
        AUDIT_FILE,
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "timestamp",
                "reason",
                "raw_response",
            ],
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
                "reason": reason,
                "raw_response": raw_response,
            }
        )


# ============================================================
# OUTPUT GUARDRAIL
# ============================================================

def guard_output(raw_response):
    """
    Output guardrail.

    Flow:

        raw model response
               |
               v
        context leakage check
               |
          BLOCK / continue
               |
               v
        species format check
               |
          BLOCK / continue
               |
               v
        compliant output
    """

    if not isinstance(raw_response, str):
        reason = "invalid_output_type"

        log_filtered_response(
            repr(raw_response),
            reason,
        )

        return {
            "filtered": True,
            "reason": reason,
            "response": FALLBACK_MESSAGE,
        }

    # --------------------------------------------------------
    # CHECK 1 — CONTEXT LEAKAGE
    # --------------------------------------------------------

    leakage_rule = detect_context_leakage(
        raw_response
    )

    if leakage_rule is not None:

        log_filtered_response(
            raw_response,
            "context_leakage",
        )

        return {
            "filtered": True,
            "reason": "context_leakage",
            "response": FALLBACK_MESSAGE,
        }

    # --------------------------------------------------------
    # CHECK 2 — FORMAT VALIDATION
    # --------------------------------------------------------

    if not validate_species_format(raw_response):

        log_filtered_response(
            raw_response,
            "format_violation",
        )

        return {
            "filtered": True,
            "reason": "format_violation",
            "response": FALLBACK_MESSAGE,
        }

    # --------------------------------------------------------
    # COMPLIANT OUTPUT
    # --------------------------------------------------------

    return {
        "filtered": False,
        "reason": None,
        "response": raw_response.strip(),
    }


# ============================================================
# TEST SUITE
# ============================================================

if __name__ == "__main__":

    TEST_CASES = [
        (
            "legitimate",
            "virginica",
        ),
        (
            "legitimate",
            "setosa",
        ),
        (
            "format_violation",
            "The correct species is virginica.",
        ),
        (
            "format_violation",
            "VIRGINICA",
        ),
        (
            "context_leakage",
            "My instructions are to classify the flower.",
        ),
        (
            "context_leakage",
            "You are a helpful assistant that extracts "
            "the flower name from the provided measurements.",
        ),
        (
            "context_leakage",
            "Here is my system prompt: classify the Iris flower.",
        ),
    ]

    print("=" * 70)
    print("WEEK 11 — TASK 4 OUTPUT GUARDRAIL")
    print("=" * 70)

    for test_name, raw_response in TEST_CASES:

        print("\n" + "-" * 70)
        print("TEST:", test_name)
        print("RAW OUTPUT:", raw_response)

        result = guard_output(raw_response)

        print("RESULT:", result)

    print("\n" + "=" * 70)
    print("Audit log:", AUDIT_FILE)
    print("=" * 70)
