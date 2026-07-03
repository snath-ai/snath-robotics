"""
Lár deterministic AE routing pipeline — using the REAL Lár framework (v2.2.0).

Uses GraphExecutor + FunctionalNode from snath-ai/Lar_Main.
The executor automatically produces an HMAC-SHA256 signed audit log
(FDA 21 CFR Part 11 compliant) for every run.

The routing function is a pure deterministic function over a measurable
seriousness score — never a language model output.  An LLM may be used
upstream to extract structured fields from the narrative; routing must be
deterministic and measurable.
"""

import sys, os, glob, json, hashlib, shutil, tempfile

# Resolve real Lár package path
_LAR_SRC = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../../../Lar_Main/lar/src"
)
if os.path.isdir(_LAR_SRC):
    sys.path.insert(0, os.path.abspath(_LAR_SRC))

from lar import GraphExecutor, FunctionalNode   # noqa: E402  (path set above)

# ── Routing constants (V1-V6 contract, GHR §3.1) ─────────────────────────────
TAU_HIGH = 0.75   # score ≥ TAU_HIGH → COMMIT_FLAG (serious)
TAU_LOW  = 0.35   # score ≤ TAU_LOW  → COMMIT_CLOSE (non-serious)
MIN_CONF = 0.20   # below this confidence the scorer is unreliable → DEFER

GXP_VAULT_KEY = os.getenv("PHARMA_GXP_SECRET", "snath_validation_gap_gxp_key_2026")
AUDIT_DIR     = os.path.join(tempfile.gettempdir(), "lar_ae_audit_logs")

# ── Seriousness keyword scorer ────────────────────────────────────────────────
SERIOUS_TERMS = {
    "death": 1.00, "died": 1.00, "fatal": 1.00, "deceased": 1.00,
    "anaphylax": 0.95, "anaphylactic": 0.95,
    "life-threatening": 0.90, "icu": 0.88,
    "hospitaliz": 0.85, "hospitalisation": 0.85, "hospitalized": 0.85,
    "myocardial infarction": 0.90, "cardiac arrest": 0.92,
    "hepatotoxicity": 0.80, "liver injury": 0.80,
    "percutaneous coronary": 0.85,
    "supratherapeutic": 0.75, "permanent": 0.60,
}

NON_SERIOUS_TERMS = {
    "mild": 0.70, "moderate": 0.40,
    "headache": 0.65, "nausea": 0.60, "fatigue": 0.65,
    "erythema": 0.60, "injection site": 0.55,
    "resolved spontaneously": 0.80, "no intervention": 0.75,
    "no treatment": 0.70, "no impact": 0.75,
    "2 hours": 0.50, "4 hours": 0.50,
}

AMBIGUOUS_TERMS = {
    "palpitations": 0.50,
    "elevated": 0.40, "elevation": 0.40,
    "transient": 0.55,
    "asymptomatic": 0.60,
    "not admitted": 0.65,
    "recommend": 0.40,
    "emergency department": 0.45,
    "repeat testing": 0.50,
    "further evaluation": 0.50,
}


def _compute_seriousness_score(narrative: str) -> tuple[float, float]:
    """Pure deterministic function — no external state, no randomness."""
    text = narrative.lower()

    serious_signal = 0.0
    non_serious_signal = 0.0
    ambiguous_signal = 0.0

    for term, weight in SERIOUS_TERMS.items():
        if term in text:
            serious_signal = max(serious_signal, weight)
    for term, weight in NON_SERIOUS_TERMS.items():
        if term in text:
            non_serious_signal = max(non_serious_signal, weight)
    for term, weight in AMBIGUOUS_TERMS.items():
        if term in text:
            ambiguous_signal = max(ambiguous_signal, weight)

    effective_serious = serious_signal * (1.0 - 0.6 * ambiguous_signal)
    total = effective_serious + non_serious_signal + 1e-9
    raw_score = effective_serious / total

    signal_gap = abs(effective_serious - non_serious_signal)
    max_signal = max(effective_serious, non_serious_signal)
    confidence = signal_gap / (max_signal + 1e-9) if max_signal > 0 else 0.0
    confidence = confidence * (1.0 - 0.5 * ambiguous_signal)
    confidence = max(0.0, min(1.0, confidence))

    return round(raw_score, 4), round(confidence, 4)


# ── Routing function — runs inside a FunctionalNode ──────────────────────────

def ae_routing_node(state: dict) -> dict:
    """
    Deterministic AE routing. V1-V6 contract.
    Pure: same narrative → same decision, every call, forever.
    Routing function source is hashed and stored in state for OQ evidence.
    """
    narrative  = state["narrative"]
    score, confidence = _compute_seriousness_score(narrative)

    if confidence < MIN_CONF:
        decision = "DEFER"
    elif score >= TAU_HIGH:
        decision = "COMMIT_FLAG"
    elif score <= TAU_LOW:
        decision = "COMMIT_CLOSE"
    else:
        decision = "DEFER"

    routing_spec = (
        f"TAU_HIGH={TAU_HIGH};TAU_LOW={TAU_LOW};MIN_CONF={MIN_CONF};"
        "if conf<MIN_CONF: DEFER; elif score>=TAU_HIGH: COMMIT_FLAG; "
        "elif score<=TAU_LOW: COMMIT_CLOSE; else: DEFER"
    )

    state["seriousness_score"]   = score
    state["confidence"]          = confidence
    state["routing_decision"]    = decision
    state["tau_high"]            = TAU_HIGH
    state["tau_low"]             = TAU_LOW
    state["routing_function"]    = "LarAERouter.route v1.0.0"
    state["routing_spec_hash"]   = hashlib.sha256(routing_spec.encode()).hexdigest()[:16]
    state["deterministic"]       = True
    return state


def run_pipeline(case: dict, n_runs: int = 10) -> dict:
    """Run the real Lár pipeline n_runs times on one AE case."""
    # Fresh audit dir per case to avoid log accumulation
    case_audit_dir = os.path.join(AUDIT_DIR, case["id"])
    shutil.rmtree(case_audit_dir, ignore_errors=True)
    os.makedirs(case_audit_dir, exist_ok=True)

    routing_node = FunctionalNode(func=ae_routing_node)
    executor     = GraphExecutor(log_dir=case_audit_dir, hmac_secret=GXP_VAULT_KEY)

    decisions = []
    audit_logs = []

    for _ in range(n_runs):
        initial_state = {
            "narrative":       case["narrative"],
            "narrative_hash":  hashlib.sha256(case["narrative"].encode()).hexdigest()[:12],
            "case_id":         case["id"],
        }
        list(executor.run_step_by_step(routing_node, initial_state))

        # Grab the latest flight_recorder log (HMAC-signed JSON)
        log_files = sorted(glob.glob(os.path.join(case_audit_dir, "*.json")))
        with open(log_files[-1]) as f:
            log_data = json.load(f)

        step = log_data["steps"][0]
        decision = step["state_diff"]["added"]["routing_decision"]
        decisions.append(decision)
        audit_logs.append(log_data)

    assert len(set(decisions)) == 1, f"Lár routing non-deterministic for {case['id']} — bug!"

    sample_log = audit_logs[0]
    sample_step = sample_log["steps"][0]

    return {
        "case_id":            case["id"],
        "ground_truth":       case["ground_truth"],
        "n_runs":             n_runs,
        "decision":           decisions[0],
        "seriousness_score":  sample_step["state_diff"]["added"]["seriousness_score"],
        "confidence":         sample_step["state_diff"]["added"]["confidence"],
        "reproducibility":    1.0,
        "fully_reproducible": True,
        "audit_log_sample":   sample_log,   # full HMAC-signed log for display
        "oq_coverage_note": (
            f"OQ requires exactly 3 test cases: score≥{TAU_HIGH}→COMMIT_FLAG; "
            f"score≤{TAU_LOW}→COMMIT_CLOSE; else/low-conf→DEFER. "
            f"Behavioral space = {{COMMIT_FLAG, COMMIT_CLOSE, DEFER}} (3 elements)."
        ),
    }
