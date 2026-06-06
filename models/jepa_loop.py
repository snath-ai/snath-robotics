"""
Snath Robotics — JEPA Learning Loop.
======================================
Wires the JEPA predictor into the D_hard queue and DMN to form a
fully closed, annotation-free learning loop.

Every timestep:
  1. Predictor computes D_pred = 1 - cos(f(z_vision), z_proprio)
  2. If D_pred > threshold: auto-determine winner from prediction geometry
  3. Log to DHardQueue with winner already set (no human needed)
  4. Every consolidate_every steps: DMN trains LoRA adapters + predictor retrained

The winner determination is self-supervised:
  D_pred high + D high  → physics conflict (ice slip, wet floor)
                           → proprio was right → winner = "proprio"
                           → LoRA corrects vision encoder
  D_pred high + D low   → sensor drift (routing didn't catch it)
                           → vision is stable reference → winner = "vision"
                           → LoRA corrects proprio encoder
  conf_vision low        → vision obscured → proprio is ground truth
                           → failure_class = "hardware_structural"

This is LeCun's JEPA claim end-to-end: the world model generates the
learning signal, the DMN encodes what was learned into LoRA adapters,
and the predictor retrained on better representations catches harder
cases in the next cycle.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple, List

import torch

log = logging.getLogger(__name__)


def _auto_winner(
    d_pred:       float,
    d:            float,
    conf_vision:  float,
    conf_proprio: float,
    tau_low:      float = 0.25,
    pred_thresh:  float = 0.30,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Self-supervised winner and failure class from prediction error geometry.
    No labels. No human. Physics provides the signal.

    Returns:
        (winner, failure_class) or (None, None) if not anomalous.

    Logic:
      D_pred ≤ threshold:         not anomalous — skip
      D_pred > thresh + D > τ_low + conf_vision low:
                                  visual system obscured → proprio ground truth
                                  winner="proprio", class="hardware_structural"
      D_pred > thresh + D > τ_low: physics conflict (ice, wet surface)
                                  → proprio was right
                                  winner="proprio", class="environmental_transient"
      D_pred > thresh + D ≤ τ_low: sensor drift (gradual, routing missed it)
                                  → vision is stable reference
                                  winner="vision", class="sensor_drift"
    """
    if d_pred <= pred_thresh:
        return None, None

    if d > tau_low:
        if conf_vision < 0.15:
            return "proprio", "hardware_structural"
        return "proprio", "environmental_transient"
    else:
        return "vision", "sensor_drift"


class JEPALearningLoop:
    """
    Closed annotation-free learning loop for Snath Robotics.

    Usage:
        loop = JEPALearningLoop(predictor, queue, dmn, router)

        # Every timestep:
        d_pred = loop.step(z_vision, z_proprio,
                           d=result.divergence,
                           conf_vision=result.conf_vision,
                           conf_proprio=result.conf_proprio,
                           scenario_id="ice_slip")

        # After a session (or automatically every consolidate_every steps):
        built = loop.consolidate()

    Args:
        predictor:         JEPAPredictor instance.
        queue:             DHardQueue instance.
        dmn:               RoboticsDMN instance.
        tau_low:           Router tau_low (for auto-winner logic).
        pred_threshold:    D_pred above this → anomalous, log event.
        consolidate_every: Run DMN cycle every N steps.
        retrain_epochs:    Predictor retraining epochs each cycle.
    """

    def __init__(
        self,
        predictor,
        queue,
        dmn,
        tau_low:           float = 0.25,
        pred_threshold:    float = 0.30,
        consolidate_every: int   = 16,
        retrain_epochs:    int   = 100,
    ):
        self.predictor         = predictor
        self.queue             = queue
        self.dmn               = dmn
        self.tau_low           = tau_low
        self.pred_threshold    = pred_threshold
        self.consolidate_every = consolidate_every
        self.retrain_epochs    = retrain_epochs

        self._step_count   = 0
        self._buf_vision:  List[torch.Tensor] = []
        self._buf_proprio: List[torch.Tensor] = []

    def step(
        self,
        z_vision:     torch.Tensor,
        z_proprio:    torch.Tensor,
        d:            float,
        conf_vision:  float,
        conf_proprio: float,
        scenario_id:  str = "",
    ) -> float:
        """
        Run one timestep of the learning loop.

        Computes D_pred, logs anomalies to DHardQueue with auto-winner,
        accumulates training buffer, and triggers consolidation every
        consolidate_every steps.

        Returns:
            d_pred: float prediction error for this timestep.
        """
        from dhard import RoboticsDHardEvent

        with torch.no_grad():
            d_pred = float(
                self.predictor.prediction_error(
                    z_vision.unsqueeze(0), z_proprio.unsqueeze(0)
                ).item()
            )

        # Accumulate for predictor retraining
        self._buf_vision.append(z_vision.detach().cpu())
        self._buf_proprio.append(z_proprio.detach().cpu())

        # Auto-winner: self-supervised, no labels
        winner, failure_class = _auto_winner(
            d_pred, d, conf_vision, conf_proprio,
            tau_low=self.tau_low, pred_thresh=self.pred_threshold,
        )

        if winner is not None:
            # Log with winner already set → immediately "resolved" → DMN can use it
            decision_str = (
                "STRUCTURAL_IMPASSE" if d >= 0.60 else "TRIGGER_REPLAN"
            )
            ev = RoboticsDHardEvent(
                z_vision      = z_vision.tolist(),
                z_proprio     = z_proprio.tolist(),
                divergence    = d,
                decision      = decision_str,
                failure_class = failure_class,
                scenario_id   = scenario_id,
                winner        = winner,       # auto-set — no human
            )
            self.queue.push(ev)

        self._step_count += 1
        if self._step_count % self.consolidate_every == 0:
            self._consolidate()

        return d_pred

    def _consolidate(self) -> List[dict]:
        """DMN consolidation + predictor retraining. Runs automatically."""
        log.info(f"[JEPALoop] consolidation at step {self._step_count}")

        # 1. DMN: cluster D_hard events → LoRA adapters
        built = self.dmn.consolidate(min_events=4, verbose=False)
        if built:
            log.info(f"[JEPALoop] DMN built {len(built)} adapter(s): "
                     f"{[m['failure_class'] for m in built]}")

        # 2. Retrain predictor on accumulated buffer
        if len(self._buf_vision) >= 8:
            from models.jepa_predictor import train_predictor
            z_vis = torch.stack(self._buf_vision)
            z_prp = torch.stack(self._buf_proprio)
            stats = train_predictor(
                self.predictor, z_vis, z_prp,
                n_epochs=self.retrain_epochs, batch_size=min(32, len(z_vis)),
            )
            log.info(f"[JEPALoop] predictor retrained: "
                     f"error {stats['error_before']:.4f} → {stats['error_after']:.4f}")

        return built

    def force_consolidate(self) -> List[dict]:
        """Trigger DMN + predictor retraining immediately."""
        return self._consolidate()

    def stats(self) -> dict:
        q = self.queue.stats()
        return {
            "steps":         self._step_count,
            "buffer_size":   len(self._buf_vision),
            "dhard_total":   q["total"],
            "dhard_resolved": q["resolved"],
            "by_class":      q.get("by_class", {}),
        }
