import math
from typing import Tuple
from vis_forecast.globals import WIND_DIRECTION_SCORE

from pydantic import BaseModel

class VulnerableWindDirection(BaseModel):
    direction: list[int]  # List of 2 integer numbers representing wind direction



def _normalize_angle(angle: float) -> float:
    """Wrap angles to the [0, 360) interval."""
    return angle % 360


def _angle_in_range(angle: float, start: float, end: float) -> bool:
    """Return True when angle falls inside the inclusive arc from start to end."""
    angle = _normalize_angle(angle)
    start = _normalize_angle(start)
    end = _normalize_angle(end)

    if start < end:
        return start <= angle <= end

    return angle >= start or angle <= end


def _minimal_angular_difference(a: float, b: float) -> float:
    """Return the signed smallest difference from b to a within [-180, 180]."""
    diff = (a - b + 180.0) % 360.0 - 180.0
    return diff if diff != -180.0 else 180.0


def wind_direction_score(
    vulnerable_wind_direction: Tuple[float, float], wind_direction: float
) -> float:
    """
    Score wind direction alignment with a vulnerable beach sector.

    Returns values in [-1, 1] where 1 means the wind blows directly into the beach
    (within the provided sector) and -1 represents an offshore wind.
    """
    start, end = vulnerable_wind_direction
    start = _normalize_angle(start)
    end = _normalize_angle(end)
    span = (end - start) % 360.0

    center = _normalize_angle(start + span / 2.0)
    delta = abs(_minimal_angular_difference(wind_direction, center))
    inside_range = _angle_in_range(wind_direction, start, end)

    half_span = span / 2.0
    if inside_range:
        raw_score = 1.0 - (delta / half_span if half_span else 0.0)
    else:
        outside_delta = delta - half_span
        denominator = 180.0 - half_span
        if denominator <= 0:
            raw_score = -1.0
        else:
            raw_score = -outside_delta / denominator
    
    for raw_score_upper_boundary, score in WIND_DIRECTION_SCORE.items():
        if raw_score <= raw_score_upper_boundary:
            return score


    
