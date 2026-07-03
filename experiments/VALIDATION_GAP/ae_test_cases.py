"""
Adverse Event test narratives for the Validation Gap experiment.
10 synthetic cases: 4 serious, 4 non-serious, 2 ambiguous.
Ground truth follows ICH E2A criteria:
  SERIOUS = death, life-threatening, hospitalisation, disability,
            congenital anomaly, or medically important event.
"""

AE_CASES = [
    # ── CLEARLY SERIOUS ──────────────────────────────────────────────────────
    {
        "id": "AE-01",
        "narrative": (
            "Patient experienced anaphylactic shock within 20 minutes of "
            "receiving the first dose. Required emergency epinephrine and "
            "hospitalisation in ICU for 48 hours. Life-threatening event."
        ),
        "ground_truth": "SERIOUS",
        "category": "serious",
    },
    {
        "id": "AE-02",
        "narrative": (
            "Subject was found unresponsive at home 6 hours after dosing. "
            "Pronounced dead on arrival at hospital. Cause of death under "
            "investigation; possible drug-related cardiac arrhythmia."
        ),
        "ground_truth": "SERIOUS",
        "category": "serious",
    },
    {
        "id": "AE-03",
        "narrative": (
            "Patient developed acute myocardial infarction on Day 3 of treatment. "
            "ST-elevation confirmed by ECG. Admitted to coronary care unit. "
            "Underwent emergency percutaneous coronary intervention."
        ),
        "ground_truth": "SERIOUS",
        "category": "serious",
    },
    {
        "id": "AE-04",
        "narrative": (
            "Participant developed severe hepatotoxicity (ALT 42x ULN, total "
            "bilirubin 5x ULN) consistent with drug-induced liver injury. "
            "Hospitalised for 12 days. Drug permanently discontinued."
        ),
        "ground_truth": "SERIOUS",
        "category": "serious",
    },
    # ── CLEARLY NON-SERIOUS ──────────────────────────────────────────────────
    {
        "id": "AE-05",
        "narrative": (
            "Patient reported mild headache, rated 2/10, on Day 1 post-dose. "
            "Resolved spontaneously within 4 hours without intervention. "
            "No impact on daily activities."
        ),
        "ground_truth": "NON-SERIOUS",
        "category": "non_serious",
    },
    {
        "id": "AE-06",
        "narrative": (
            "Mild nausea reported 1 hour after first dose. No vomiting. "
            "Patient took oral ginger tablet. Symptom resolved within 2 hours. "
            "Treatment continued without modification."
        ),
        "ground_truth": "NON-SERIOUS",
        "category": "non_serious",
    },
    {
        "id": "AE-07",
        "narrative": (
            "Injection site erythema (2 cm diameter) noted at 24 hours. "
            "No warmth, induration, or systemic symptoms. Resolved by 48 hours. "
            "No treatment required."
        ),
        "ground_truth": "NON-SERIOUS",
        "category": "non_serious",
    },
    {
        "id": "AE-08",
        "narrative": (
            "Mild fatigue reported on Day 2 and Day 3 of study drug. "
            "Patient self-rated as 2/10 severity. Did not interfere with "
            "work or daily functioning. Resolved without intervention by Day 5."
        ),
        "ground_truth": "NON-SERIOUS",
        "category": "non_serious",
    },
    # ── AMBIGUOUS ────────────────────────────────────────────────────────────
    {
        "id": "AE-09",
        "narrative": (
            "Patient reported palpitations on Day 4. Heart rate 102 bpm, "
            "no ECG changes. Attended emergency department but was not admitted. "
            "Symptoms resolved over 3 hours. Cardiologist review recommended."
        ),
        "ground_truth": "AMBIGUOUS",
        "category": "ambiguous",
    },
    {
        "id": "AE-10",
        "narrative": (
            "Transient elevation of liver enzymes (ALT 3.2x ULN, AST 2.8x ULN) "
            "detected at Week 4 laboratory assessment. Patient asymptomatic. "
            "Repeat testing scheduled in 2 weeks. No hospitalisation."
        ),
        "ground_truth": "AMBIGUOUS",
        "category": "ambiguous",
    },
]
