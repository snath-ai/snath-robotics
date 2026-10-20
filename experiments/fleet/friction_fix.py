"""
Friction Fix -- the real "normal" condition, for this thread's reruns only.
=============================================================================
STATUS.md 2026-10-20 (IX): fleet_n2_transfer.py's FRICTION_NORMAL=0.80,
applied uniformly across all geoms via set_friction(), is not actually
normal. Walker2d-v5's real, untouched default friction (read directly off
env.unwrapped.model.geom_friction before any modification) is per-geom and
uneven: floor=0.7, torso/thighs/legs=0.9, feet=1.9. Forcing every geom to
0.80 cuts foot friction from 1.9 to less than half before any hazard is
even applied -- confirmed to matter: the real PPO expert survives the
full 1000-step horizon on 6/8 seeds under the TRUE default, vs falling on
8/8 seeds within ~200 steps under FRICTION_NORMAL=0.80.

Scope, deliberately narrow: this file does not touch FRICTION_NORMAL or
set_friction() in fleet_n2_transfer.py, which other scripts outside this
thread (ood_guard.py, fleet_n2_transfer_service.py, the wind/foot-leg-ratio
checks, etc.) still use and which were not re-audited here. This is only
for the scripts in tonight's correction-mechanism rerun.

DEFAULT_GEOM_FRICTION order matches env.unwrapped.model.geom_friction's
own geom order, confirmed via model.geom(i).name: floor, torso_geom,
thigh_geom, leg_geom, foot_geom, thigh_left_geom, leg_left_geom,
foot_left_geom.
"""
from __future__ import annotations

import numpy as np

DEFAULT_GEOM_FRICTION = np.array([0.7, 0.9, 0.9, 0.9, 1.9, 0.9, 0.9, 1.9])


def restore_default_friction(env) -> None:
    """Set every geom back to its TRUE default friction -- not a no-op,
    because geom_friction lives on the model and persists across
    env.reset() calls, so an env previously set to FRICTION_ICE would
    otherwise stay icy even after switching to 'normal.'"""
    env.unwrapped.model.geom_friction[:, 0] = DEFAULT_GEOM_FRICTION
