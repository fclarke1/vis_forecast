# vulnerable wind directions: moving clockwise, angles between hit directly onto the beach (allowing waves to form)
LOCATIONS = {
    "falmouth": {"latitude": 50.152573, "longitude": -5.066270, "vulnerable_wind_direction": [80, 210]},
    "towan_beach": {"latitude": 50.157088, "longitude": -4.984155, "vulnerable_wind_direction": [45, 230]},
}

# using relative wind direction - give score
WIND_DIRECTION_SCORE = {
    -0.5: -1,
    -0.2: -0.5,
    0: 0,
    0.1: 0.5,
    1: 1.0
}