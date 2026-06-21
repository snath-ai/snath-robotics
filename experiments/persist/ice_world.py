"""
IceWorld: A custom MuJoCo environment for testing physics assumption violation
detection, adaptive response verification, and bounded persistent adaptation.

Part of the PERSIST paper (Paper 6 in the Snath series):
  Physics-grounded Iterative Refinement with Scope-bounded Incremental
  Stopping Threshold

Environment structure:
  A corridor with four sequential zones:
    Zone 1: Ice patch          — zero friction
    Zone 2: Ice + slope        — friction loss + gravity component change
    Zone 3: Force perturbation — sudden lateral/longitudinal force (wind/impact)
    Zone 4: Novel combined     — all three simultaneously (never seen in training)

Robot: Hopper (simple, clean action space, interventions are readable)

The divergence signal (from PAV) runs continuously as the robot moves.
The persistence loop (PERSIST) uses divergence as the fitness function
for iterative response refinement.

Zones are defined by x-position thresholds along the corridor.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Zone definitions (x-position thresholds along corridor)
# ---------------------------------------------------------------------------

ZONES = {
    "normal":    (0.0,  2.0),   # baseline terrain
    "ice":       (2.0,  4.0),   # Zone 1: zero friction
    "ice_slope": (4.0,  6.0),   # Zone 2: ice + slope (gravity component)
    "force":     (6.0,  8.0),   # Zone 3: lateral force perturbation
    "novel":     (8.0, 10.0),   # Zone 4: all three combined
}

ZONE_FRICTION = {
    "normal":    1.0,
    "ice":       0.005,  # nearly frictionless
    "ice_slope": 0.005,
    "force":     1.0,
    "novel":     0.005,
}

ZONE_GRAVITY_OFFSET = {
    "normal":    0.0,
    "ice":       0.0,
    "ice_slope": -5.0,   # steep downhill slope component (m/s^2)
    "force":     0.0,
    "novel":     -5.0,
}

ZONE_LATERAL_FORCE = {
    "normal":    0.0,
    "ice":       0.0,
    "ice_slope": 0.0,
    "force":     20.0,   # strong lateral wind force (N)
    "novel":     20.0,
}


# ---------------------------------------------------------------------------
# IceWorld environment
# ---------------------------------------------------------------------------

class IceWorldEnv(gym.Env):
    """
    IceWorld wraps Hopper-v4 with zone-based physics modification.

    Key additions over standard Hopper:
      - get_current_zone(): returns active zone name from x-position
      - divergence_signal: rolling divergence from baseline proprioceptive stats
      - zone_log: list of (step, zone, divergence) tuples for plotting
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        baseline_steps: int = 200,
        divergence_window: int = 20,
        escalation_threshold: float = 3.0,
        render_mode=None,
    ):
        """
        Args:
            baseline_steps:        Steps on normal terrain to build baseline stats.
            divergence_window:     Rolling window size for divergence computation.
            escalation_threshold:  Divergence magnitude above which escalation fires.
            render_mode:           'human' or 'rgb_array'.
        """
        self.render_mode = render_mode
        self.baseline_steps = baseline_steps
        self.divergence_window = divergence_window
        self.escalation_threshold = escalation_threshold

        self._env = gym.make("Hopper-v5", render_mode=render_mode)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

        # Baseline proprioceptive statistics (built during normal zone)
        self._baseline_obs: list = []
        self._baseline_mean: np.ndarray | None = None
        self._baseline_std:  np.ndarray | None = None
        self._baseline_built = False

        # Rolling observation window for live divergence
        self._obs_window: list = []

        # State
        self._step_count = 0
        self._x_position = 0.0
        self._current_zone = "normal"

        # Logs
        self.zone_log:       list[tuple] = []   # (step, zone, divergence)
        self.escalation_log: list[tuple] = []   # (step, zone, reason)

    # -----------------------------------------------------------------------
    # Zone utilities
    # -----------------------------------------------------------------------

    def get_current_zone(self) -> str:
        for zone, (lo, hi) in ZONES.items():
            if lo <= self._x_position < hi:
                return zone
        return "novel" if self._x_position >= 8.0 else "normal"

    def _apply_zone_physics(self, action: np.ndarray) -> np.ndarray:
        """
        Apply zone physics by modifying MuJoCo model parameters directly.
        geom[0] = floor, body[1] = torso.
        """
        zone = self._current_zone
        u = self._env.unwrapped

        # Floor sliding friction
        u.model.geom_friction[0, 0] = ZONE_FRICTION[zone]

        # Slope: x-axis gravity component (downhill tilt)
        u.model.opt.gravity[0] = ZONE_GRAVITY_OFFSET[zone]

        # Lateral wind force on torso
        lateral = ZONE_LATERAL_FORCE[zone]
        u.data.xfrc_applied[1, 0] = lateral

        return action

    # -----------------------------------------------------------------------
    # Divergence signal
    # -----------------------------------------------------------------------

    def _update_baseline(self, obs: np.ndarray):
        if not self._baseline_built:
            self._baseline_obs.append(obs)
            if len(self._baseline_obs) >= self.baseline_steps:
                arr = np.array(self._baseline_obs)
                self._baseline_mean = arr.mean(axis=0)
                self._baseline_std  = arr.std(axis=0) + 1e-8
                self._baseline_built = True

    def compute_divergence(self, obs: np.ndarray) -> float:
        """
        Compute divergence of current observation from baseline.
        Returns scalar divergence magnitude (z-score norm over window).
        Returns 0.0 if baseline not yet built.
        """
        if not self._baseline_built:
            return 0.0

        self._obs_window.append(obs)
        if len(self._obs_window) > self.divergence_window:
            self._obs_window.pop(0)

        window_mean = np.array(self._obs_window).mean(axis=0)
        z = (window_mean - self._baseline_mean) / self._baseline_std
        return float(np.linalg.norm(z))

    def is_escalation_needed(self, divergence: float) -> bool:
        return divergence >= self.escalation_threshold

    # -----------------------------------------------------------------------
    # Gym interface
    # -----------------------------------------------------------------------

    def reset(self, seed=None, options=None, keep_baseline: bool = False):
        """
        Args:
            keep_baseline: If True, baseline statistics are preserved across the
                           reset so they can accumulate across multiple episodes.
        """
        # Restore default MuJoCo physics before each episode
        u = self._env.unwrapped
        u.model.geom_friction[0, 0] = 1.0
        u.model.opt.gravity[0] = 0.0
        u.data.xfrc_applied[:] = 0.0

        obs, info = self._env.reset(seed=seed, options=options)
        self._step_count = 0
        self._x_position = 0.0
        self._current_zone = "normal"

        if not keep_baseline:
            self._baseline_obs.clear()
            self._baseline_built = False

        self._obs_window.clear()
        self.zone_log.clear()
        self.escalation_log.clear()
        return obs, info

    def step(self, action: np.ndarray):
        modified_action = self._apply_zone_physics(action)
        obs, reward, terminated, truncated, info = self._env.step(modified_action)

        # Update x-position estimate from obs (Hopper obs[0] = z-height,
        # obs[5] = x-velocity; integrate x-velocity for position)
        x_vel = float(obs[5]) if len(obs) > 5 else 0.0
        self._x_position += x_vel * 0.008   # dt = 0.008s for Hopper
        self._current_zone = self.get_current_zone()

        # Build baseline on normal terrain
        if self._current_zone == "normal":
            self._update_baseline(obs)

        # Compute divergence
        divergence = self.compute_divergence(obs)

        # Log
        self.zone_log.append((self._step_count, self._current_zone, divergence))

        # Check escalation
        if self.is_escalation_needed(divergence) and self._current_zone != "normal":
            self.escalation_log.append((
                self._step_count,
                self._current_zone,
                f"divergence={divergence:.3f} >= threshold={self.escalation_threshold}"
            ))

        info["zone"]       = self._current_zone
        info["divergence"] = divergence
        info["x_position"] = self._x_position
        info["escalation"] = self.is_escalation_needed(divergence)

        self._step_count += 1
        return obs, reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("IceWorld smoke test")
    print("=" * 50)

    env = IceWorldEnv(baseline_steps=50, escalation_threshold=2.5)
    obs, _ = env.reset(seed=42)

    divergence_by_zone = {z: [] for z in ZONES}

    for step in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        zone = info["zone"]
        div  = info["divergence"]
        divergence_by_zone[zone].append(div)

        if info["escalation"]:
            print(f"  Step {step:3d} | Zone: {zone:12s} | Divergence: {div:.3f} | ESCALATION")

        if terminated or truncated:
            obs, _ = env.reset()

    print("\nMean divergence per zone:")
    for zone, divs in divergence_by_zone.items():
        if divs:
            print(f"  {zone:12s}: {np.mean(divs):.3f}")

    print(f"\nEscalation events: {len(env.escalation_log)}")
    env.close()
    print("Done.")
