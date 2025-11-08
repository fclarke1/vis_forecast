from typing import Iterable, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.axes import Axes
from vis_forecast.globals import LOCATIONS, WIND_SPEED_SCORE, RAIN_PROB_SCORE, TIME_WEIGHT_FUNCTION

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


def wind_score(
    vulnerable_wind_direction: Tuple[float, float], wind_direction: float, wind_speed: float
) -> float:
    """
    Score wind direction and wind speed score.

    Returns values in [0, 1]
    """
    start, end = vulnerable_wind_direction
    start = _normalize_angle(start)
    end = _normalize_angle(end)
    is_vulnerable_wind_direction = _angle_in_range(wind_direction, start, end)

    wind_score = 0
    if is_vulnerable_wind_direction:
        for wind_speed_boundary, score in WIND_SPEED_SCORE["onshore"].items():
            if wind_speed <= wind_speed_boundary:
                wind_score = score
                break
    else:
        for wind_speed_boundary, score in WIND_SPEED_SCORE["offshore"].items():
            if wind_speed <= wind_speed_boundary:
                wind_score = score
                break
    return wind_score
            


def rain_prob_score(prob_rain: float, prob_heavy_rain: float):
    """
    Give a score between [0,1] depending on the probability of rain.
    
    Return the max score for either rain type
    """
    for prob_rain_upper_boundary, prob_rain_score in RAIN_PROB_SCORE["prob_rain"].items():
        if prob_rain <= prob_rain_upper_boundary:
            score_1 = prob_rain_score
            break
    for prob_heavy_rain_upper_boundary, prob_heavy_rain_score in RAIN_PROB_SCORE["prob_heavy_rain"].items():
        if prob_heavy_rain <= prob_heavy_rain_upper_boundary:
            score_2 = prob_heavy_rain_score
            break
    score = max(score_1, score_2)
    return score



def time_weighting(time_delta: pd.Timedelta | np.ndarray | pd.Series) -> float | np.ndarray | pd.Series:
    """
    Compute a cosine weight in [0, 1] for the supplied time delta.

    The delta is normalised against the configured history window (in days)
    before being mapped through a cosine curve, so that recent timestamps
    receive a weight close to 1 and those at the edge of the window approach 0.
    """
    history_days = TIME_WEIGHT_FUNCTION["history_days"]
    if history_days <= 0:
        raise ValueError("TIME_WEIGHT_FUNCTION['history_days'] must be positive.")

    history_window = pd.Timedelta(days=history_days)
    # Convert any timedelta-like input to pandas Timedelta/TimedeltaIndex for division.
    delta = pd.to_timedelta(time_delta)
    normalized = delta / history_window

    normalized = np.clip(np.asarray(normalized, dtype=float), 0.0, 1.0)
    weights = 0.5 * (np.cos(np.pi * normalized) + 1.0)

    # Preserve scalar return type for scalar input.
    if weights.ndim == 0:
        return float(weights)
    return weights

def visualize_rain_prob_scores(
    prob_rain_values: Iterable[float] | None = None,
    prob_heavy_rain_values: Iterable[float] | None = None,
    cmap: str = "viridis",
) -> None:
    """
    Plot a heatmap of the rain probability score across probability combinations.

    Defaults to 5 percentage point steps covering [0, 100].
    """
    if prob_rain_values is None:
        prob_rain_values = np.arange(0, 101, 5)
    if prob_heavy_rain_values is None:
        prob_heavy_rain_values = np.arange(0, 101, 5)

    prob_rain_values = np.sort(np.asarray(list(prob_rain_values), dtype=float))
    prob_heavy_rain_values = np.sort(np.asarray(list(prob_heavy_rain_values), dtype=float))

    if prob_rain_values.size == 0 or prob_heavy_rain_values.size == 0:
        raise ValueError("Probability sequences must not be empty.")

    score_grid = np.zeros((prob_heavy_rain_values.size, prob_rain_values.size))
    for i, heavy_prob in enumerate(prob_heavy_rain_values):
        for j, rain_prob in enumerate(prob_rain_values):
            score_grid[i, j] = rain_prob_score(rain_prob, heavy_prob)

    fig, ax = plt.subplots()
    heatmap = ax.imshow(
        score_grid,
        origin="lower",
        extent=[
            prob_rain_values.min(),
            prob_rain_values.max(),
            prob_heavy_rain_values.min(),
            prob_heavy_rain_values.max(),
        ],
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=1,
    )
    ax.set_title("Rain Probability Score")
    ax.set_xlabel("Probability of rain (%)")
    ax.set_ylabel("Probability of heavy rain (%)")

    # Use evenly spaced tick labels to avoid overcrowding the axis.
    for axis_values, axis in ((prob_rain_values, ax.xaxis), (prob_heavy_rain_values, ax.yaxis)):
        if axis_values.size <= 12:
            axis.set_ticks(axis_values)
        else:
            axis.set_ticks(np.linspace(axis_values.min(), axis_values.max(), 11))

    fig.colorbar(heatmap, ax=ax, label="Score")
    fig.tight_layout()
    plt.show()
    
    
    
def visualize_wind_scores(
    vulnerable_wind_direction: Tuple[float, float] | None = None,
    wind_direction_values: Iterable[float] | None = None,
    wind_speed_values: Iterable[float] | None = None,
    cmap: str = "plasma",
) -> None:
    """
    Plot a heatmap of the wind score across wind direction and speed combinations.

    Defaults cover 0–355° in 5° steps and 0–30 units of wind speed.
    """
    if vulnerable_wind_direction is None:
        vulnerable_wind_direction = tuple(LOCATIONS["falmouth"]["vulnerable_wind_direction"])

    if wind_direction_values is None:
        wind_direction_values = np.arange(0, 360, 5)
    if wind_speed_values is None:
        wind_speed_values = np.arange(0, 31, 1)

    wind_direction_values = np.unique(np.asarray(list(wind_direction_values), dtype=float))
    wind_speed_values = np.unique(np.asarray(list(wind_speed_values), dtype=float))

    if wind_direction_values.size == 0 or wind_speed_values.size == 0:
        raise ValueError("Wind direction and speed sequences must not be empty.")

    score_grid = np.zeros((wind_speed_values.size, wind_direction_values.size))
    for i, speed in enumerate(wind_speed_values):
        for j, direction in enumerate(wind_direction_values):
            score_grid[i, j] = wind_score(vulnerable_wind_direction, direction, speed)

    fig, ax = plt.subplots()
    heatmap = ax.imshow(
        score_grid,
        origin="lower",
        extent=[
            wind_direction_values.min(),
            wind_direction_values.max(),
            wind_speed_values.min(),
            wind_speed_values.max(),
        ],
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=1,
    )
    ax.set_title("Wind Score")
    ax.set_xlabel("Wind direction (degrees)")
    ax.set_ylabel("Wind speed")

    # Mark the vulnerable wind sector to clarify the highest-sensitivity region.
    sector_start, sector_end = vulnerable_wind_direction
    sector_start = _normalize_angle(sector_start)
    sector_end = _normalize_angle(sector_end)
    x_min, x_max = wind_direction_values.min(), wind_direction_values.max()
    if sector_start <= sector_end:
        ax.axvspan(sector_start, sector_end, color="white", alpha=0.12)
    else:
        ax.axvspan(sector_start, x_max, color="white", alpha=0.12)
        ax.axvspan(x_min, sector_end, color="white", alpha=0.12)

    for axis_values, axis in ((wind_direction_values, ax.xaxis), (wind_speed_values, ax.yaxis)):
        if axis_values.size <= 12:
            axis.set_ticks(axis_values)
        else:
            axis.set_ticks(np.linspace(axis_values.min(), axis_values.max(), 11))

    fig.colorbar(heatmap, ax=ax, label="Score")
    fig.tight_layout()
    plt.show()


def _fill_vulnerable_sector(ax: Axes, vulnerable_wind_direction: Tuple[float, float], radius: float) -> None:
    """Shade the vulnerable wind sector on a polar axis."""
    sector_start, sector_end = vulnerable_wind_direction
    sector_start = _normalize_angle(sector_start)
    sector_end = _normalize_angle(sector_end)

    def _fill(theta_start: float, theta_end: float) -> None:
        theta = np.linspace(theta_start, theta_end, 200)
        ax.fill_between(theta, 0, radius, color="white", alpha=0.12)

    theta_start = np.deg2rad(sector_start)
    theta_end = np.deg2rad(sector_end)
    if theta_start <= theta_end:
        _fill(theta_start, theta_end)
    else:
        _fill(theta_start, 2 * np.pi)
        _fill(0, theta_end)


def visualize_wind_scores_polar(
    vulnerable_wind_direction: Tuple[float, float] | None = None,
    wind_direction_values: Iterable[float] | None = None,
    wind_speed_values: Iterable[float] | None = None,
    cmap: str = "plasma",
) -> None:
    """
    Plot a polar heatmap of the wind score across wind direction and speed combinations.

    Direction wraps around 360°, with radius corresponding to wind speed.
    """
    if vulnerable_wind_direction is None:
        vulnerable_wind_direction = tuple(LOCATIONS["falmouth"]["vulnerable_wind_direction"])

    if wind_direction_values is None:
        wind_direction_values = np.arange(0, 360, 5)
    if wind_speed_values is None:
        wind_speed_values = np.arange(0, 31, 1)

    wind_direction_values = np.unique(np.asarray(list(wind_direction_values), dtype=float))
    wind_speed_values = np.sort(np.unique(np.asarray(list(wind_speed_values), dtype=float)))

    if wind_direction_values.size == 0 or wind_speed_values.size == 0:
        raise ValueError("Wind direction and speed sequences must not be empty.")

    score_grid = np.zeros((wind_speed_values.size, wind_direction_values.size))
    for i, speed in enumerate(wind_speed_values):
        for j, direction in enumerate(wind_direction_values):
            score_grid[i, j] = wind_score(vulnerable_wind_direction, direction, speed)

    # Close the circle by repeating the first column at 360°.
    theta_degrees = np.append(wind_direction_values, wind_direction_values[0] + 360)
    theta_radians = np.deg2rad(theta_degrees)
    score_extended = np.hstack([score_grid, score_grid[:, :1]])

    theta_mesh, radius_mesh = np.meshgrid(theta_radians, wind_speed_values)

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    heatmap = ax.pcolormesh(
        theta_mesh,
        radius_mesh,
        score_extended,
        cmap=cmap,
        vmin=0,
        vmax=1,
        shading="auto",
    )

    ax.set_title("Wind Score (Polar)")
    ax.set_ylim(0, wind_speed_values.max())
    ax.set_yticks(wind_speed_values if wind_speed_values.size <= 10 else np.linspace(0, wind_speed_values.max(), 6))
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))

    _fill_vulnerable_sector(ax, vulnerable_wind_direction, wind_speed_values.max())

    fig.colorbar(heatmap, ax=ax, pad=0.1, label="Score")
    fig.tight_layout()
    plt.show()


def time_weighting(time_delta: pd.Timedelta | np.ndarray | pd.Series) -> float | np.ndarray | pd.Series:
    """
    Compute a cosine weight in [0, 1] for the supplied time delta.
    The delta is normalised against the configured history window (in days)
    before being mapped through a cosine curve, so that recent timestamps
    receive a weight close to 1 and those at the edge of the window approach 0.
    """
    history_days = TIME_WEIGHT_FUNCTION["history_days"]
    if history_days <= 0:
        raise ValueError("TIME_WEIGHT_FUNCTION['history_days'] must be positive.")
    history_window = pd.Timedelta(days=history_days)
    
    # Convert any timedelta-like input to pandas Timedelta/TimedeltaIndex for division.
    delta = pd.to_timedelta(time_delta)
    normalized = delta / history_window
    normalized = np.clip(np.asarray(normalized, dtype=float), 0.0, 1.0)
    weights = 0.5 * (np.cos(np.pi * normalized) + 1.0)
    
    # Preserve scalar return type for scalar input.
    if weights.ndim == 0:
        return float(weights)
    return weights


def create_forecast(data_met: pd.DataFrame) -> pd.DataFrame:
    """
    Given weather df with score - compute a time weighted rolling average for each 
    """
    history_days = TIME_WEIGHT_FUNCTION["history_days"]
    history_window = pd.Timedelta(days=history_days)
    now = pd.Timestamp.now(tz=data_met["time"].dt.tz if data_met["time"].dt.tz is not None else None)
    relevant_time = now - history_window
    mask_relevant_time = data_met["time"] > relevant_time
    df_forecast = data_met.loc[mask_relevant_time, ["time", "date_creation", "location_name",
                                                    "score", "score_wind", "score_rain"]].copy()

    def _time_weighted_score(time: pd.Timestamp, location: str, df: pd.DataFrame) -> float:
        """
        Compute the weighted average score for a given time and location
        based on the past 7 days of scores in df.
        """
        window_start = time - pd.Timedelta(days=history_days)
        mask = (df["location_name"] == location) & (df["time"].between(window_start, time))
        df_window = df.loc[mask, ["time", "score"]]
        if df_window.empty:
            return np.nan
        deltas = time - df_window["time"]
        weights = time_weighting(deltas)
        return np.average(df_window["score"], weights=weights)
    
    # Apply row-wise, passing df_forecast as context
    df_forecast["vis_score"] = df_forecast.apply(
        lambda row: _time_weighted_score(row["time"], row["location_name"], df_forecast),
        axis=1
    )
    return df_forecast

