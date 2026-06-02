"""
Shared type definitions for Snath Robotics.
"""
from enum import Enum


class RouteDecision(str, Enum):
    COMMIT_TRAJECTORY  = "COMMIT_TRAJECTORY"   # streams agree — proceed
    TRIGGER_REPLAN     = "TRIGGER_REPLAN"       # recoverable contradiction — adapt
    STRUCTURAL_IMPASSE = "STRUCTURAL_IMPASSE"   # irreconcilable — safe fallback
    DEFER              = "DEFER"                # one stream uncertain — lean on confident arm
