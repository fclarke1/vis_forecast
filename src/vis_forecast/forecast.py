import pandas as pd
from pydantic import BaseModel

class VulnerableWindDirection(BaseModel):
    direction: list[int]  # List of 2 integer numbers representing wind direction



def wind_direction_score(vulnerable_wind_direction:list[int], wind_direction:int)->float:
    """ check wind direction if it is in vulnerable wind direction, and give a soft score """
    is_includes_north = vulnerable_wind_direction[0] > vulnerable_wind_direction[1]
    if is_includes_north