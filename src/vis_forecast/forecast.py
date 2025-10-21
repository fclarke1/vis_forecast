import pandas as pd
from pydantic import BaseModel
from vis_forecast.globals import WIND_DIRECTION_SCORE

class VulnerableWindDirection(BaseModel):
    direction: list[int]  # List of 2 integer numbers representing wind direction



def wind_direction_score(vulnerable_wind_direction:list[int], wind_direction:int)->float:
    """ check wind direction if it is in vulnerable wind direction, and give a soft score """
    bad_angle = vulnerable_wind_direction.copy()
    wind_direction_copy = wind_direction
    print(f"Start: badangle: {bad_angle} - wind_direction: {wind_direction}")
    if bad_angle[0] > bad_angle[1]:
        bad_angle[0] -= 360
    
    if wind_direction > bad_angle[1]:
        wind_direction -= 360
    print(f"wrap around: badangle: {bad_angle} - wind_direction: {wind_direction}")
    
    if wind_direction > bad_angle[0] and wind_direction < bad_angle[1]:
        direct_bad_angle = (bad_angle[0] + bad_angle[1]) / 2
        # gives symmetric relative direction - 0 on edge, 1 is direct at beach
        relative_wind_direction = 1 - abs((((wind_direction - bad_angle[0]) / (bad_angle[1] - bad_angle[0])) * 2) - 1)
        print(f"inside BAD angle: badangle: {bad_angle} - wind_direction: {wind_direction} - relative wind: {relative_wind_direction} - direct_bad_angle: {direct_bad_angle}")
    else:
        # ensure wind direction is between bad_angle - then apply the same above but make it negative
        if wind_direction < bad_angle[0]:
            bad_angle[1] -= 360
        else:
            bad_angle[0] += 360
        direct_good_angle = (bad_angle[0] + bad_angle[1]) / 2
        relative_wind_direction = -1 * (1 - abs((((wind_direction - bad_angle[1]) / (bad_angle[0] - bad_angle[1])) * 2) - 1))
        print(f"inside GOOD angle: badangle: {bad_angle} - wind_direction: {wind_direction} - relative wind: {relative_wind_direction} - direct_good_angle: {direct_good_angle}")
        
    wind_score = 10
    for direction_boundary, boundary_score in WIND_DIRECTION_SCORE.items():
        print(f"wind score iteration: boundary direction: {direction_boundary}")
        if relative_wind_direction <= direction_boundary:
            wind_score = boundary_score
            break
    return wind_score


    