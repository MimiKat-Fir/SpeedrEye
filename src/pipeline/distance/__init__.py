"""Distance estimation methods used by the SpeedrEye pipeline."""

from .direct import DirectDistanceEstimator
from .geometry_guided import GeometryGuidedDistanceEstimator


def build_distance_estimator(method, detector, config):
    if method == "direct":
        return DirectDistanceEstimator(detector, config)
    if method == "geometry":
        return GeometryGuidedDistanceEstimator(detector, config)
    if method == "none":
        return None
    raise ValueError(f"Unknown distance method: {method}")


__all__ = [
    "DirectDistanceEstimator",
    "GeometryGuidedDistanceEstimator",
    "build_distance_estimator",
]
