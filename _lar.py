"""
_lar.py — locate the Lár-JEPA engine for Snath Robotics.

Resolution order:
  1. $SNATH_ROBOTICS_LARJEPA  — set this env var to point anywhere
  2. ../Lar-JEPA/lar_jepa     — sibling directory (standard install)
  3. ../lar_jepa               — dev fallback
  4. ~/lar_jepa                — home install
"""
import os
import sys

_CANDIDATES = [
    os.environ.get("SNATH_ROBOTICS_LARJEPA"),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Lar-JEPA", "lar_jepa")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lar_jepa")),
    os.path.expanduser("~/lar_jepa"),
]

for _p in _CANDIDATES:
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
        break
