"""
PERSIST: Persistence Loop

The core new invariant in Paper 6.

Biological principle:
  A child encountering ice does not fall, log the fall, and wait.
  It falls, adjusts, tries again, finds what works, remembers it.
  When unsure which approach to try, it quickly trials a few and
  commits to whichever is reducing the slip fastest.

Formal definition:
  Given a divergence signal D(t) and a response policy R(delta),
  the persistence loop iteratively increments delta until either:
    (a) D(t) returns to within baseline bounds  -> success, store adaptor
    (b) delta reaches maximum safe bound         -> escalate, ask for help
    (c) Steps exceed patience budget             -> escalate, log stream

The divergence signal that detected the violation is the same signal
that verifies whether the response worked. One stream. Two purposes.

Six components:
  1. Tournament    -- trial top-K candidates, commit to fastest divergence reducer
  2. Increment     -- increase delta until divergence normalises
  3. Compose       -- orthogonal residual triggers second adaptor
  4. Scope         -- delta_max or patience exceeded -> escalate
  5. Escalate      -- log stream as curriculum, ask for help
  6. Consolidate   -- store successful response in memory

Six phases (mirroring biological learning):
  Phase 1: Encounter   -- no adaptor, divergence fires, log stream
  Phase 2: First try   -- adaptor trained, try delta, verify via divergence
  Phase 3: Composition -- residual divergence triggers second adaptor
  Phase 4: Exhaustion  -- novel scenario, scope exceeded, escalate
  Phase 5: Memory      -- stored adaptor retrieved, zero failed attempts
  Phase 6: Tournament  -- ambiguous signatures, trial K candidates, pick winner
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# Adaptor
# ---------------------------------------------------------------------------

@dataclass
class Adaptor:
    """
    A stored response to a specific divergence class.

    name:           Human-readable label (e.g. 'ice', 'slope')
    delta:          Corrective action offset (array, same shape as action space)
    divergence_signature: Mean divergence pattern this adaptor was trained on
    success_count:  Number of times this adaptor successfully resolved divergence
    """
    name: str
    delta: np.ndarray
    divergence_signature: np.ndarray
    success_count: int = 0


# ---------------------------------------------------------------------------
# Adaptor Library
# ---------------------------------------------------------------------------

class AdaptorLibrary:
    """
    Memory store for adaptors. Matches incoming divergence signature
    to closest stored adaptor via cosine similarity.
    Corresponds to EIM's world-grounded memory retrieval.
    """

    def __init__(self, similarity_threshold: float = 0.75):
        self._adaptors: list[Adaptor] = []
        self.similarity_threshold = similarity_threshold

    def store(self, adaptor: Adaptor):
        self._adaptors.append(adaptor)
        print(f"  [Library] Stored adaptor: '{adaptor.name}'")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return -1.0
        return float(np.dot(a, b) / (na * nb))

    def retrieve(self, divergence_signature: np.ndarray) -> Adaptor | None:
        """
        Return closest adaptor if similarity >= threshold, else None.
        None means novel scenario — no prior experience.
        """
        if not self._adaptors:
            return None

        best_sim = -1.0
        best_adaptor = None

        for adaptor in self._adaptors:
            sim = self._cosine_similarity(divergence_signature,
                                          adaptor.divergence_signature)
            if sim > best_sim:
                best_sim = sim
                best_adaptor = adaptor

        if best_sim >= self.similarity_threshold:
            print(f"  [Library] Retrieved: '{best_adaptor.name}' "
                  f"(similarity={best_sim:.3f})")
            return best_adaptor

        print(f"  [Library] No match (best={best_sim:.3f} "
              f"< threshold={self.similarity_threshold})")
        return None

    def retrieve_top_k(
        self,
        divergence_signature: np.ndarray,
        k: int = 3,
        band: float = 0.15,
    ) -> list[tuple[Adaptor, float]]:
        """
        Return top-K adaptors within a similarity band of the best match.
        Used by AdaptorTournament when signatures are ambiguous.

        Args:
            divergence_signature: incoming divergence pattern
            k:    maximum candidates to return
            band: include candidates within `band` of the best similarity score

        Returns:
            List of (adaptor, similarity) sorted descending. Empty if no match
            reaches the library threshold.
        """
        if not self._adaptors:
            return []

        scored = [
            (a, self._cosine_similarity(divergence_signature, a.divergence_signature))
            for a in self._adaptors
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        best_sim = scored[0][1]
        if best_sim < self.similarity_threshold:
            return []

        candidates = [
            (a, s) for a, s in scored
            if s >= best_sim - band
        ][:k]

        print(f"  [Library] Top-{k} candidates: "
              + ", ".join(f"'{a.name}'({s:.3f})" for a, s in candidates))
        return candidates

    def __len__(self):
        return len(self._adaptors)


# ---------------------------------------------------------------------------
# Persistence Loop
# ---------------------------------------------------------------------------

@dataclass
class PersistResult:
    """Outcome of one persistence loop run."""
    phase: str
    success: bool
    steps_to_resolution: int
    final_divergence: float
    adaptors_used: list[str]
    escalated: bool
    divergence_curve: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Adaptor Tournament
# ---------------------------------------------------------------------------

@dataclass
class TournamentResult:
    """Outcome of one tournament run."""
    winner: Adaptor | None
    rates: dict           # adaptor_name -> reduction_rate (D/step)
    trial_steps_used: int
    divergence_curves: dict  # adaptor_name -> list[float]


class AdaptorTournament:
    """
    When multiple adaptors have similar cosine similarity scores,
    commit to the one that reduces divergence fastest — not the one
    that looks most similar on paper.

    Biological principle:
      When uncertain which gait works on ice, try a shuffle for 5 steps,
      a crouch for 5 steps, arms-out for 5 steps. Whichever stops the
      slipping fastest — commit to it. Don't guess, let the body tell you.

    Mechanism:
      For each candidate adaptor, apply it for `trial_steps` steps and
      measure the divergence reduction rate:

        rate(a) = (D_entry - D_after_trial) / trial_steps

      Winner = argmax rate(a).

      If all rates are negative (every candidate made things worse),
      return None — no candidate is helping, trigger escalation.

    Note:
      Actions in physical systems are irreversible — the robot cannot
      reset to the same state between trials. Trials are sequential.
      State drift between trials is accepted as a physical reality.
      The rate signal is robust to small state drift because it is
      measuring divergence reduction, not absolute divergence level.
    """

    def __init__(self, trial_steps: int = 5):
        """
        Args:
            trial_steps: Steps to trial each candidate before scoring.
                         Small enough to be fast, large enough for signal.
                         Empirically 3-7 steps is sufficient for Hopper.
        """
        self.trial_steps = trial_steps

    def run(
        self,
        candidates: list[tuple[Adaptor, float]],
        obs: np.ndarray,
        divergence_entry: float,
        base_action_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        step_fn: Callable[[np.ndarray], tuple[np.ndarray, float]],
        delta_init: float = 1.0,
    ) -> TournamentResult:
        """
        Trial each candidate adaptor for `trial_steps` steps.
        Return the winner (highest divergence reduction rate).

        Args:
            candidates:        List of (adaptor, similarity) from retrieve_top_k.
            obs:               Current observation at tournament entry.
            divergence_entry:  Divergence magnitude at tournament entry.
            base_action_fn:    fn(obs, delta) -> action
            step_fn:           fn(action) -> (next_obs, divergence)
            delta_init:        Initial delta scale for all candidates.

        Returns:
            TournamentResult with winner and per-candidate rates.
        """
        print(f"\n  [Tournament] {len(candidates)} candidates | "
              f"trial_steps={self.trial_steps} | "
              f"entry_divergence={divergence_entry:.3f}")

        rates = {}
        curves = {}
        current_obs = obs.copy()
        current_div = divergence_entry

        for adaptor, sim in candidates:
            delta = delta_init * adaptor.delta
            trial_start_div = current_div   # divergence at the start of THIS trial
            trial_curve = [current_div]

            for _ in range(self.trial_steps):
                action = base_action_fn(current_obs, delta)
                current_obs, current_div = step_fn(action)
                trial_curve.append(current_div)

            # Rate: measure this candidate's own marginal contribution, not cumulative
            rate = (trial_start_div - current_div) / self.trial_steps
            rates[adaptor.name] = rate
            curves[adaptor.name] = trial_curve

            print(f"    '{adaptor.name}': rate={rate:+.4f}/step  "
                  f"(sim={sim:.3f}, D_entry={trial_start_div:.3f}, D_final={current_div:.3f})")

        # Winner = highest reduction rate
        if not rates:
            return TournamentResult(
                winner=None, rates=rates,
                trial_steps_used=self.trial_steps,
                divergence_curves=curves,
            )

        best_name = max(rates, key=rates.__getitem__)
        best_rate  = rates[best_name]

        # If best rate is negative or zero — every candidate made things worse
        if best_rate <= 0.0:
            print(f"  [Tournament] All candidates non-reducing. No winner.")
            return TournamentResult(
                winner=None, rates=rates,
                trial_steps_used=self.trial_steps * len(candidates),
                divergence_curves=curves,
            )

        winner = next(a for a, _ in candidates if a.name == best_name)
        print(f"  [Tournament] Winner: '{winner.name}' "
              f"(rate={best_rate:+.4f}/step)")

        return TournamentResult(
            winner=winner,
            rates=rates,
            trial_steps_used=self.trial_steps * len(candidates),
            divergence_curves=curves,
        )


class PersistenceLoop:
    """
    The PERSIST core mechanism.

    Six-component loop:
      1. Tournament    — trial top-K candidates, commit to fastest reducer
      2. Increment     — increase delta until divergence normalises
      3. Compose       — orthogonal residual triggers second adaptor
      4. Scope         — delta_max or patience exceeded → escalate
      5. Escalate      — log stream as curriculum, ask for help
      6. Consolidate   — store successful response in memory

    Given:
      - A divergence function D(obs) -> float
      - An action function A(obs, delta) -> action  (base policy + delta)
      - An environment step function step(action) -> (obs, divergence)
      - An adaptor library

    The loop:
      1. Retrieve closest adaptor from library (or None if novel)
      2. Apply adaptor delta
      3. Measure divergence after application
      4. If divergence normalised -> success
      5. If not -> increment delta, retry
      6. If residual divergence suggests second dimension -> compose second adaptor
      7. If max delta reached or patience exhausted -> escalate
      8. On success -> update adaptor success count, store if new
    """

    def __init__(
        self,
        library: AdaptorLibrary,
        action_dim: int,
        delta_init: float = 0.05,
        delta_increment: float = 0.05,
        delta_max: float = 0.5,
        normalisation_threshold: float = 0.8,
        escalation_threshold: float = 3.0,
        patience: int = 30,
        composition_residual_threshold: float = 1.2,
        tournament_k: int = 3,
        tournament_band: float = 0.15,
        tournament_trial_steps: int = 5,
    ):
        """
        Args:
            library:                      Adaptor memory store.
            action_dim:                   Dimension of action space.
            delta_init:                   Starting corrective offset magnitude.
            delta_increment:              How much to increase delta per failed attempt.
            delta_max:                    Maximum corrective offset before escalation.
            normalisation_threshold:      Divergence below this = resolved.
            escalation_threshold:         Divergence above this = escalate immediately.
            patience:                     Max steps before escalation regardless.
            composition_residual_threshold: Residual divergence that triggers
                                           second adaptor retrieval.
            tournament_k:                 Max candidates to trial in tournament.
            tournament_band:              Similarity band for tournament candidacy.
            tournament_trial_steps:       Steps per candidate in tournament trial.
        """
        self.library = library
        self.action_dim = action_dim
        self.delta_init = delta_init
        self.delta_increment = delta_increment
        self.delta_max = delta_max
        self.normalisation_threshold = normalisation_threshold
        self.escalation_threshold = escalation_threshold
        self.patience = patience
        self.composition_residual_threshold = composition_residual_threshold
        self.tournament = AdaptorTournament(trial_steps=tournament_trial_steps)
        self.tournament_k = tournament_k
        self.tournament_band = tournament_band

    def run(
        self,
        obs: np.ndarray,
        divergence_signature: np.ndarray,
        current_divergence: float,
        base_action_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        step_fn: Callable[[np.ndarray], tuple[np.ndarray, float]],
        zone: str = "unknown",
        phase_label: str = "",
    ) -> PersistResult:
        """
        Run one persistence loop episode.

        Args:
            obs:                   Current observation.
            divergence_signature:  Divergence pattern vector for library lookup.
            current_divergence:    Divergence magnitude at loop entry.
            base_action_fn:        fn(obs, delta) -> action
            step_fn:               fn(action) -> (next_obs, divergence)
            zone:                  Current zone name for logging.
            phase_label:           Phase label for logging.

        Returns:
            PersistResult
        """

        print(f"\n{'='*60}")
        print(f"  PERSIST | Zone: {zone} | Phase: {phase_label}")
        print(f"  Entry divergence: {current_divergence:.3f}")
        print(f"{'='*60}")

        divergence_curve = [current_divergence]
        adaptors_used = []
        step_count = 0

        # Immediate escalation if divergence already extreme
        if current_divergence >= self.escalation_threshold:
            print(f"  ESCALATE immediately — divergence {current_divergence:.3f} "
                  f">= threshold {self.escalation_threshold}")
            return PersistResult(
                phase=phase_label or "escalation",
                success=False,
                steps_to_resolution=0,
                final_divergence=current_divergence,
                adaptors_used=[],
                escalated=True,
                divergence_curve=divergence_curve,
            )

        # --- Step 1: Retrieve adaptor (with tournament if ambiguous) ---
        candidates = self.library.retrieve_top_k(
            divergence_signature,
            k=self.tournament_k,
            band=self.tournament_band,
        )

        if not candidates:
            # Novel scenario — no prior experience
            print("  Novel scenario. No adaptor found.")
            print("  Logging stream as curriculum for future adaptor.")
            return PersistResult(
                phase="encounter",
                success=False,
                steps_to_resolution=0,
                final_divergence=current_divergence,
                adaptors_used=[],
                escalated=True,
                divergence_curve=divergence_curve,
            )

        if len(candidates) > 1:
            # Tournament: multiple candidates — trial each, pick fastest reducer
            t_result = self.tournament.run(
                candidates=candidates,
                obs=obs,
                divergence_entry=current_divergence,
                base_action_fn=base_action_fn,
                step_fn=step_fn,
                delta_init=self.delta_init,
            )
            step_count += t_result.trial_steps_used
            for curve in t_result.divergence_curves.values():
                divergence_curve.extend(curve[1:])

            if t_result.winner is None:
                # Every candidate made things worse — escalate
                print("  Tournament: no winning adaptor. Escalating.")
                return PersistResult(
                    phase="escalation",
                    success=False,
                    steps_to_resolution=step_count,
                    final_divergence=divergence_curve[-1],
                    adaptors_used=[a.name for a, _ in candidates],
                    escalated=True,
                    divergence_curve=divergence_curve,
                )
            adaptor = t_result.winner
            # Update current divergence to post-tournament value
            current_divergence = divergence_curve[-1]
            # Early exit if tournament already resolved it
            if current_divergence <= self.normalisation_threshold:
                adaptor.success_count += 1
                print(f"  Resolved during tournament in {step_count} steps.")
                return PersistResult(
                    phase="tournament",
                    success=True,
                    steps_to_resolution=step_count,
                    final_divergence=current_divergence,
                    adaptors_used=[adaptor.name],
                    escalated=False,
                    divergence_curve=divergence_curve,
                )
        else:
            adaptor = candidates[0][0]

        adaptors_used.append(adaptor.name)
        current_delta = self.delta_init * adaptor.delta

        # --- Step 2: Persistence loop ---
        composed = False
        second_adaptor = None

        for attempt in range(self.patience):
            action = base_action_fn(obs, current_delta)
            obs, divergence = step_fn(action)
            divergence_curve.append(divergence)
            step_count += 1

            print(f"  Attempt {attempt+1:2d} | delta_mag={np.linalg.norm(current_delta):.3f} "
                  f"| divergence={divergence:.3f}")

            # Success
            if divergence <= self.normalisation_threshold:
                adaptor.success_count += 1
                adaptor.resolved_delta = current_delta.copy()   # exact correction for memory
                print(f"  RESOLVED in {step_count} steps. "
                      f"Adaptors used: {adaptors_used}")
                return PersistResult(
                    phase=phase_label or ("composition" if composed else "first_try"),
                    success=True,
                    steps_to_resolution=step_count,
                    final_divergence=divergence,
                    adaptors_used=adaptors_used,
                    escalated=False,
                    divergence_curve=divergence_curve,
                )

            # Immediate escalation threshold
            if divergence >= self.escalation_threshold:
                print(f"  ESCALATE — divergence {divergence:.3f} exceeded threshold")
                return PersistResult(
                    phase=phase_label or "escalation",
                    success=False,
                    steps_to_resolution=step_count,
                    final_divergence=divergence,
                    adaptors_used=adaptors_used,
                    escalated=True,
                    divergence_curve=divergence_curve,
                )

            # Composition trigger — residual divergence suggests second dimension
            if (not composed
                    and divergence >= self.composition_residual_threshold
                    and attempt >= 3):
                print(f"  Residual divergence {divergence:.3f} — trying composition...")

                # Compute orthogonal residual: remove the component of the
                # divergence signature already explained by the first adaptor.
                # This ensures retrieval targets a DIFFERENT dimension.
                sig_a = adaptor.divergence_signature
                norm_sq = np.dot(sig_a, sig_a)
                if norm_sq > 1e-8:
                    projection = (np.dot(divergence_signature, sig_a) / norm_sq) * sig_a
                    residual_sig = divergence_signature - projection
                else:
                    residual_sig = divergence_signature.copy()

                # Only retrieve if residual has meaningful magnitude
                if np.linalg.norm(residual_sig) > 0.1:
                    second_adaptor = self.library.retrieve(residual_sig)
                    if second_adaptor and second_adaptor.name != adaptor.name:
                        current_delta = current_delta + self.delta_init * second_adaptor.delta
                        adaptors_used.append(second_adaptor.name)
                        composed = True
                        print(f"  Composed with '{second_adaptor.name}'")
                        continue
                    else:
                        print(f"  No distinct second adaptor found for residual.")

            # Increment delta
            current_delta_mag = np.linalg.norm(current_delta)
            if current_delta_mag >= self.delta_max:
                print(f"  Delta max reached ({current_delta_mag:.3f}). ESCALATING.")
                return PersistResult(
                    phase=phase_label or "escalation",
                    success=False,
                    steps_to_resolution=step_count,
                    final_divergence=divergence,
                    adaptors_used=adaptors_used,
                    escalated=True,
                    divergence_curve=divergence_curve,
                )

            # Increment
            current_delta = current_delta + self.delta_increment * adaptor.delta

        # Patience exhausted
        print(f"  Patience exhausted after {step_count} steps. ESCALATING.")
        return PersistResult(
            phase=phase_label or "escalation",
            success=False,
            steps_to_resolution=step_count,
            final_divergence=divergence_curve[-1],
            adaptors_used=adaptors_used,
            escalated=True,
            divergence_curve=divergence_curve,
        )


# ---------------------------------------------------------------------------
# Smoke test — five phases
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    random.seed(42)
    np.random.seed(42)

    action_dim = 3
    library = AdaptorLibrary(similarity_threshold=0.70)

    # --- Seed library with ice and slope adaptors ---
    ice_sig   = np.array([1.0, 0.2, 0.1])
    slope_sig = np.array([0.1, 1.0, 0.3])

    library.store(Adaptor(
        name="ice",
        delta=np.array([-0.3, -0.1, 0.0]),   # reduce forward torque
        divergence_signature=ice_sig,
    ))
    library.store(Adaptor(
        name="slope",
        delta=np.array([0.1, -0.2, 0.1]),    # compensate gravity
        divergence_signature=slope_sig,
    ))

    loop = PersistenceLoop(
        library=library,
        action_dim=action_dim,
        delta_init=1.0,
        delta_increment=0.1,
        delta_max=1.0,
        normalisation_threshold=0.5,
        escalation_threshold=3.5,
        patience=20,
        composition_residual_threshold=1.0,
    )

    # Simulate base_action_fn and step_fn
    def base_action_fn(obs, delta):
        return np.clip(obs[:action_dim] + delta, -1, 1)

    def make_step_fn(target_divergence: float, decay: float = 0.15):
        """Simulates divergence decaying toward target as delta is applied."""
        state = {"div": 2.5}
        def step_fn(action):
            # Divergence decays toward target proportional to action magnitude
            effect = np.linalg.norm(action) * decay
            state["div"] = max(target_divergence, state["div"] - effect)
            obs = np.random.randn(action_dim)
            return obs, state["div"]
        return step_fn

    dummy_obs = np.random.randn(action_dim)

    print("\n" + "="*60)
    print("PHASE 2: First try (ice)")
    print("="*60)
    result2 = loop.run(
        obs=dummy_obs,
        divergence_signature=ice_sig,
        current_divergence=2.2,
        base_action_fn=base_action_fn,
        step_fn=make_step_fn(target_divergence=0.3, decay=0.2),
        zone="ice",
        phase_label="first_try",
    )
    print(f"Result: success={result2.success}, "
          f"steps={result2.steps_to_resolution}, "
          f"escalated={result2.escalated}")

    print("\n" + "="*60)
    print("PHASE 3: Composition (ice + slope)")
    print("="*60)
    combined_sig = ice_sig * 0.8 + slope_sig * 0.6
    result3 = loop.run(
        obs=dummy_obs,
        divergence_signature=combined_sig,
        current_divergence=2.0,
        base_action_fn=base_action_fn,
        step_fn=make_step_fn(target_divergence=0.2, decay=0.12),
        zone="ice_slope",
        phase_label="composition",
    )
    print(f"Result: success={result3.success}, "
          f"steps={result3.steps_to_resolution}, "
          f"adaptors={result3.adaptors_used}, "
          f"escalated={result3.escalated}")

    print("\n" + "="*60)
    print("PHASE 4: Novel — no adaptor (escalation)")
    print("="*60)
    novel_sig = np.array([0.9, 0.8, 0.9])   # doesn't match ice or slope
    novel_lib = AdaptorLibrary(similarity_threshold=0.95)   # high threshold = no match
    novel_loop = PersistenceLoop(library=novel_lib, action_dim=action_dim)
    result4 = novel_loop.run(
        obs=dummy_obs,
        divergence_signature=novel_sig,
        current_divergence=2.8,
        base_action_fn=base_action_fn,
        step_fn=make_step_fn(target_divergence=2.5, decay=0.01),
        zone="novel",
        phase_label="encounter",
    )
    print(f"Result: success={result4.success}, escalated={result4.escalated}")

    print("\n" + "="*60)
    print("PHASE 5: Memory (ice seen before — zero failed attempts)")
    print("="*60)
    result5 = loop.run(
        obs=dummy_obs,
        divergence_signature=ice_sig,
        current_divergence=2.1,
        base_action_fn=base_action_fn,
        step_fn=make_step_fn(target_divergence=0.2, decay=0.25),
        zone="ice",
        phase_label="memory",
    )
    print(f"Result: success={result5.success}, "
          f"steps={result5.steps_to_resolution}, "
          f"escalated={result5.escalated}")

    print("\n" + "="*60)
    print("PHASE 6: Tournament (ambiguous signatures — ice vs wind)")
    print("="*60)
    # Wind adaptor — similar signature to ice but different delta
    wind_sig = np.array([0.85, 0.3, 0.2])   # close to ice_sig but distinct
    library.store(Adaptor(
        name="wind",
        delta=np.array([0.0, -0.2, 0.3]),    # lateral compensation
        divergence_signature=wind_sig,
    ))

    # Ambiguous incoming signature — between ice and wind
    ambiguous_sig = np.array([0.92, 0.25, 0.15])

    # step_fn: ice adaptor resolves it (rate > wind adaptor rate)
    ice_step_fn   = make_step_fn(target_divergence=0.2, decay=0.22)
    wind_step_fn  = make_step_fn(target_divergence=1.5, decay=0.05)

    # Alternate step_fn per adaptor during tournament
    call_count = [0]
    def tournament_step_fn(action):
        call_count[0] += 1
        # First trial_steps calls = ice adaptor trial, next = wind adaptor trial
        if call_count[0] <= 5:
            return ice_step_fn(action)
        return wind_step_fn(action)

    result6 = loop.run(
        obs=dummy_obs,
        divergence_signature=ambiguous_sig,
        current_divergence=2.3,
        base_action_fn=base_action_fn,
        step_fn=tournament_step_fn,
        zone="ice",
        phase_label="tournament",
    )
    print(f"Result: success={result6.success}, "
          f"steps={result6.steps_to_resolution}, "
          f"adaptors={result6.adaptors_used}, "
          f"escalated={result6.escalated}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for label, result in [
        ("Phase 2 (first_try)",   result2),
        ("Phase 3 (composition)", result3),
        ("Phase 4 (encounter)",   result4),
        ("Phase 5 (memory)",      result5),
        ("Phase 6 (tournament)",  result6),
    ]:
        status = "SUCCESS" if result.success else ("ESCALATED" if result.escalated else "FAILED")
        print(f"  {label:28s} | {status:9s} | steps={result.steps_to_resolution:3d} "
              f"| final_div={result.final_divergence:.3f}")
