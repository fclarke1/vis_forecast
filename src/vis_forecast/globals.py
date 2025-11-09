# vulnerable wind directions: moving clockwise, angles between hit directly onto the beach (allowing waves to form)
LOCATIONS = {
    "falmouth": {"latitude": 50.152573, "longitude": -5.066270, "vulnerable_wind_direction": [80, 210]},
    "towan_beach": {"latitude": 50.157088, "longitude": -4.984155, "vulnerable_wind_direction": [45, 230]},
    "dodman_point": {"latitude": 50.222568, "longitude": -4.806776, "vulnerable_wind_direction": [45, 270]},
    "thurlestone": {"latitude": 50.263044, "longitude": -3.870057, "vulnerable_wind_direction": [160, 315]},
    "start_point": {"latitude": 50.229288, "longitude": -3.650760, "vulnerable_wind_direction": [0, 120]},
    "lannacombe": {"latitude": 50.221171, "longitude": -3.680452, "vulnerable_wind_direction": [90,260]},
    "berry_head": {"latitude": 50.402568, "longitude": -3.481143, "vulnerable_wind_direction": [270, 220]},
    "branscombe": {"latitude": 50.683979, "longitude": -3.116856, "vulnerable_wind_direction": [90, 270]},
    "isle_portland": {"latitude": 50.521858, "longitude": -2.448750, "vulnerable_wind_direction": [90,290]},
    "bude": {"latitude": 50.817853, "longitude": -4.560764, "vulnerable_wind_direction": [180,0]},
    "fishermans_cove": {"latitude": 50.238151, "longitude": -5.371382, "vulnerable_wind_direction": [310, 90]},
}

# using relative wind direction - give score
WIND_SPEED_SCORE = {
    "offshore": {
        4: 0,
        7: 0.5,
        1000: 1
    },
    "onshore": {
        7: 0,
        12: 0.25,
        15: 0.75,
        1000: 1
    }
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

TIME_WEIGHT_FUNCTION = {
    "fn": "cos",
    "history_days": 7,
}

VIS_FORECAST_RAG_SCORE = {
    0.1: "Green",
    0.3: "Amber",
    1.0: "Red"
}