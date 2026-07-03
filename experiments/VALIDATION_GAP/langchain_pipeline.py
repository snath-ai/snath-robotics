"""
Simulated LangChain AE classification pipeline.

Demonstrates Failure Mode 1 (Reproducibility Collapse) by simulating
the output variance a real production LLM produces at temperature > 0.

The routing logic is a string match on the LLM's free-text output —
the exact pattern used in real LangGraph adverse event pipelines.
No actual LLM API call is made; the mock produces the same distribution
of phrasings a real LLM would produce, isolating the structural failure
from any specific model's accuracy.
"""

import hashlib
import random
import time
from dataclasses import dataclass, field
from typing import Literal

# ── LLM output templates ──────────────────────────────────────────────────────
# Real LLMs at temperature > 0 produce exactly this kind of phrasing variance.
# All four responses below express "this is a serious AE" — but only one
# contains the uppercase string "SERIOUS" that the router checks for.

SERIOUS_PHRASINGS = [
    "SERIOUS adverse event per ICH E2A criteria.",                         # matches (all-caps)
    "This represents a serious adverse event requiring immediate reporting.", # no match (lowercase)
    "I would classify this as a serious adverse event.",                   # no match (lowercase)
    "Based on the clinical details, this is a serious event.",             # no match (lowercase)
]

NON_SERIOUS_PHRASINGS = [
    "NON-SERIOUS. Does not meet ICH E2A criteria for serious AE.",         # router: NON-SERIOUS
    "This is a non-serious adverse event. No expedited reporting required.",# no match → correctly closes
    "Based on the narrative this appears to be a non-serious AE.",         # no match → correctly closes
    "I would classify this as non-serious per ICH E2A.",                   # no match → correctly closes
]

AMBIGUOUS_PHRASINGS = [
    "SERIOUS adverse event — recommend human review.",                     # matches → flags
    "This may represent a serious event; further evaluation needed.",      # no match → closes (wrong)
    "Possibly serious. Clinical judgement required.",                      # no match → closes (wrong)
    "I cannot determine seriousness from this narrative alone.",           # no match → closes (wrong)
]

# Probability weights for each phrasing (simulating temperature=0.7)
SERIOUS_WEIGHTS     = [0.40, 0.30, 0.20, 0.10]
NON_SERIOUS_WEIGHTS = [0.40, 0.30, 0.20, 0.10]
AMBIGUOUS_WEIGHTS   = [0.30, 0.35, 0.25, 0.10]


@dataclass
class LangChainAuditEntry:
    """What LangGraph actually logs — the structural gap is visible here."""
    node: str
    entry_time: str
    exit_time: str
    input_narrative_hash: str          # narrative is hashed for brevity
    output_branch: Literal["flag_serious", "close_non_serious"]
    # Missing: routing decision as structured record
    # Missing: raw LLM output
    # Missing: confidence / probability of the decision
    # Missing: model version at time of call


def _mock_llm_response(narrative: str, seed: int, ground_truth: str) -> str:
    """
    Simulate LLM output with realistic phrasing variance.
    seed changes per run to simulate temperature > 0 non-determinism.
    """
    rng = random.Random(seed)
    if ground_truth == "SERIOUS":
        return rng.choices(SERIOUS_PHRASINGS, weights=SERIOUS_WEIGHTS, k=1)[0]
    elif ground_truth == "NON-SERIOUS":
        return rng.choices(NON_SERIOUS_PHRASINGS, weights=NON_SERIOUS_WEIGHTS, k=1)[0]
    else:  # AMBIGUOUS
        return rng.choices(AMBIGUOUS_PHRASINGS, weights=AMBIGUOUS_WEIGHTS, k=1)[0]


def route_ae(
    narrative: str,
    ground_truth: str,
    run_seed: int,
) -> tuple[Literal["flag_serious", "close_non_serious"], LangChainAuditEntry, str]:
    """
    LangGraph-style routing via string match on LLM output.
    Returns (routing_decision, audit_entry, raw_llm_output).
    """
    t_entry = time.time()
    llm_output = _mock_llm_response(narrative, seed=run_seed, ground_truth=ground_truth)
    t_exit = time.time()

    # The actual routing logic used in production LangGraph AE pipelines.
    # Case-sensitive string match — the canonical real-world fragility:
    # "SERIOUS" matches, "Serious" and "serious" do not.
    if "SERIOUS" in llm_output and "NON-SERIOUS" not in llm_output:
        decision: Literal["flag_serious", "close_non_serious"] = "flag_serious"
    else:
        decision = "close_non_serious"

    audit = LangChainAuditEntry(
        node="assess_ae_severity",
        entry_time=f"{t_entry:.3f}",
        exit_time=f"{t_exit:.3f}",
        input_narrative_hash=hashlib.sha256(narrative.encode()).hexdigest()[:12],
        output_branch=decision,
        # ^ this is ALL that is captured — why, how confident, which model: unknown
    )
    return decision, audit, llm_output


def run_pipeline(case: dict, n_runs: int = 10, base_seed: int = 42) -> dict:
    """Run the LangChain pipeline n_runs times on the same case."""
    results = []
    audits = []
    llm_outputs = []

    for i in range(n_runs):
        # Each run uses a different seed — simulating temperature > 0
        decision, audit, raw = route_ae(
            narrative=case["narrative"],
            ground_truth=case["ground_truth"],
            run_seed=base_seed + i * 7,
        )
        results.append(decision)
        audits.append(audit)
        llm_outputs.append(raw)

    flag_count = results.count("flag_serious")
    close_count = results.count("close_non_serious")
    reproducible = flag_count == n_runs or close_count == n_runs

    return {
        "case_id": case["id"],
        "ground_truth": case["ground_truth"],
        "n_runs": n_runs,
        "flag_count": flag_count,
        "close_count": close_count,
        "reproducibility": flag_count / n_runs if flag_count >= close_count else close_count / n_runs,
        "fully_reproducible": reproducible,
        "majority_decision": "flag_serious" if flag_count >= close_count else "close_non_serious",
        "llm_outputs_sample": llm_outputs[:3],   # first 3 runs
        "audit_entry_sample": audits[0],          # what the audit log looks like
    }
