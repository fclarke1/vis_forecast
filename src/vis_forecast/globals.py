# vulnerable wind directions: moving clockwise, angles between hit directly onto the beach (allowing waves to form)
LOCATIONS = {
    "falmouth": {"latitude": 50.152573, "longitude": -5.066270, "vulnerable_wind_direction": [80, 210]},
    "towan_beach": {"latitude": 50.157088, "longitude": -4.984155, "vulnerable_wind_direction": [45, 230]},
}

# using relative wind direction - give score
WIND_DIRECTION_SCORE = {
    -0.5: 0,
    -0.2: 0,
    0: 0,
    0.1: 0.5,
    1: 1.0
}

WIND_SPEED_VULNERABLE_SCORE = {
    4: 0,
    7: 0.5,
    1000: 1
}

WIND_SPEED_SCORE = {
    7: 0,
    12: 0.25,
    15: 0.75,
    1000: 1
}

# takes the maax of heavy rain and rain score, eg. prob_rain=50, and prob_heavy_rain=10 will have score prob_rain score
RAIN_PROB_SCORE = {
    "prob_rain": {
        20: 0,
        60: 0.5,
        100: 0.8
    },
    "prob_heavy_rain": {
        20: 0,
        60: 0.75,
        100: 1.0
    }
}

TIME_WEIGHT_FUNCTION = "cos"