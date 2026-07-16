"""
Shared action policies for PAV/fleet experiments.

`random`: uniform actions in [-0.4, 0.4]^6 — the PAV-paper policy.
`cpg`   : scripted periodic gait (central-pattern-generator style).
          Sinusoidal torques with alternating leg phase, per-episode
          phase/frequency jitter. A stand-in for the trained locomotion
          policy anticipated by PAV Limitation 3 / future direction 2:
          structured gait cycles -> structured proprioceptive windows,
          so a physics change (ice) reads as a cycle disruption rather
          than vanishing into policy noise.
"""
from __future__ import annotations

import numpy as np


class RandomPolicy:
    def __init__(self, rng: np.random.Generator, act_dim: int = 6):
        self.rng = rng
        self.act_dim = act_dim

    def reset(self):
        pass

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.rng.uniform(-0.4, 0.4, size=self.act_dim)


class CPGPolicy:
    """
    Walker2d-v5 action = [thigh_r, leg_r, foot_r, thigh_l, leg_l, foot_l].
    Right and left legs run pi out of phase; foot joints lead thighs by
    pi/2. Amplitude/frequency/phase jittered per episode via `rng`.
    """
    def __init__(self, rng: np.random.Generator, act_dim: int = 6,
                 base_freq: float = 0.05, base_amp: float = 0.55,
                 noise_std: float = 0.05):
        self.rng = rng
        self.act_dim = act_dim
        self.base_freq = base_freq      # cycles per env step
        self.base_amp = base_amp
        self.noise_std = noise_std
        self.t = 0
        self.reset()

    def reset(self):
        self.t = 0
        self.freq = self.base_freq * float(self.rng.uniform(0.8, 1.2))
        self.amp = self.base_amp * float(self.rng.uniform(0.85, 1.15))
        self.phase0 = float(self.rng.uniform(0, 2 * np.pi))

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        ph = 2 * np.pi * self.freq * self.t + self.phase0
        right = np.array([np.sin(ph), np.sin(ph), np.sin(ph + np.pi / 2)])
        left = np.array([np.sin(ph + np.pi), np.sin(ph + np.pi),
                         np.sin(ph + 3 * np.pi / 2)])
        a = self.amp * np.concatenate([right, left])
        a = a + self.rng.normal(0, self.noise_std, size=self.act_dim)
        self.t += 1
        return np.clip(a, -1.0, 1.0)


def make_policy(name: str, rng: np.random.Generator) -> RandomPolicy | CPGPolicy:
    if name == "random":
        return RandomPolicy(rng)
    if name == "cpg":
        return CPGPolicy(rng)
    raise ValueError(f"unknown policy: {name}")
