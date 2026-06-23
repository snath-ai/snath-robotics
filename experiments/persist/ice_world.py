"""
IceWorld: MuJoCo Hopper-v5 corridor with four physics perturbation zones.

Zone layout (x-position of robot torso):
  Normal    [0, 2)   baseline friction (μ = 1.0)
  Ice       [2, 4)   reduced friction (μ = 0.20)
  Ice+Slope [4, 6)   friction loss + gravity bias (Δg_x = +0.5 m/s²)
  Force     [6, 8)   lateral perturbation (F_x = 1.5 N on torso)
  Novel     [8, 10)  all three simultaneously

Perturbations are calibrated to be detectable (D > D_NORM) but recoverable
with a small action-delta adaptor.  Near-zero friction (0.005) prevents
convergence entirely and is experimentally non-viable.

Callers apply zones explicitly via set_zone(); x-position tracking is
available via get_x_pos() for corridor-style experiments.
"""

from __future__ import annotations
import numpy as np
import gymnasium as gym

# physics config per zone
ZONE_PARAMS: dict[str, dict] = {
    "normal":    {"friction": 1.000, "gravity_x": 0.0, "force_x": 0.0},
    "ice":       {"friction": 0.200, "gravity_x": 0.0, "force_x": 0.0},
    "ice_slope": {"friction": 0.200, "gravity_x": 0.5, "force_x": 0.0},
    "force":     {"friction": 1.000, "gravity_x": 0.0, "force_x": 1.5},
    "novel":     {"friction": 0.200, "gravity_x": 0.5, "force_x": 1.5},
}

# Hopper-v5 body ordering: world=0, torso=1, thigh=2, leg=3, foot=4
_TORSO_BODY = 1
# Window covers ~3 full hop cycles (Hopper hops at ~3 Hz, MuJoCo at 50 Hz →
# ~17 steps/cycle; 50 steps ≈ 3 cycles).  A window shorter than one cycle
# leaves the rolling mean perpetually out of phase with the baseline mean,
# inflating D(t) in normal conditions and masking perturbation signal.
_WINDOW     = 50


class IceWorld:
    """Hopper-v5 with caller-controlled zone physics.

    Usage:
        env = IceWorld(seed=42)
        obs, _ = env.reset()
        env.set_zone("ice")
        obs, rew, done, trunc, info = env.step(action)
    """

    def __init__(self, seed: int = 42) -> None:
        self.env  = gym.make("Hopper-v5")
        self.seed = seed
        self._base_friction: np.ndarray | None = None
        self.active_zone: str = "normal"

    def reset(self) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(seed=self.seed)
        if self._base_friction is None:
            self._base_friction = self.env.unwrapped.model.geom_friction.copy()
        self.set_zone("normal")
        return obs, info

    def step(self, action: np.ndarray):
        # Re-assert lateral force each step — xfrc_applied persists in MuJoCo
        # data but explicit reassignment guards against library-internal resets.
        if ZONE_PARAMS[self.active_zone]["force_x"] != 0.0:
            self.env.unwrapped.data.xfrc_applied[_TORSO_BODY, 0] = (
                ZONE_PARAMS[self.active_zone]["force_x"]
            )
        obs, rew, term, trunc, info = self.env.step(action)
        info["zone"] = self.active_zone
        return obs, rew, term, trunc, info

    def set_zone(self, zone: str) -> None:
        """Apply named-zone physics immediately; safe to call mid-episode."""
        cfg   = ZONE_PARAMS[zone]
        model = self.env.unwrapped.model
        data  = self.env.unwrapped.data
        model.geom_friction[:]    = self._base_friction.copy()
        model.geom_friction[:, 0] = cfg["friction"]
        model.opt.gravity[:]      = np.array([cfg["gravity_x"], 0.0, -9.81])
        data.xfrc_applied[:]      = 0.0
        if cfg["force_x"] != 0.0:
            data.xfrc_applied[_TORSO_BODY, 0] = cfg["force_x"]
        self.active_zone = zone

    def get_x_pos(self) -> float:
        return float(self.env.unwrapped.data.qpos[0])

    def close(self) -> None:
        self.env.close()

    @property
    def obs_dim(self) -> int:
        return int(self.env.observation_space.shape[0])

    @property
    def act_dim(self) -> int:
        return int(self.env.action_space.shape[0])


def compute_divergence(
    obs: np.ndarray,
    buf: list[np.ndarray],
    mu: np.ndarray,
    sigma: np.ndarray,
) -> float:
    """D(t) = ‖(rolling_mean(buf[-W:]) − μ) / σ‖₂   (PERSIST paper eq. 4)."""
    buf.append(obs.copy())
    if len(buf) > _WINDOW:
        buf.pop(0)
    rolling = np.mean(buf, axis=0)
    return float(np.linalg.norm((rolling - mu) / (sigma + 1e-8)))
