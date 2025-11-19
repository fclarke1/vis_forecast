import requests
from pathlib import Path
import pandas as pd
import os
import json
from loguru import logger

from vis_forecast.globals import LOCATIONS, VIS_FORECAST_RAG_SCORE
from vis_forecast.forecast import rain_prob_score, wind_score, create_forecast


class MetOfficeDataLoader:
    def __init__(self, api_key:str=None, data_dir:str="./data"):
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.environ["MET_OFFICE_API_KEY"]
        self.url = "https://data.hub.api.metoffice.gov.uk/sitespecific/v0/point/three-hourly"
        self.data_dir = Path(data_dir)
        
        self.met_data_path = self.data_dir / "met_data.csv"
        self.forecast_data_path = self.data_dir / "forecast.csv"
        self.json_data_path = self.data_dir / "forecast.json"
        self.load_data_dir()
    

    def load_data_dir(self) -> None:
        """
        Load previously recorded data from MET office. If no data is there then create fresh csv files
        """
        data = {
            "met_data": {"path": self.met_data_path},
            "forecast": {"path": self.forecast_data_path},
        }
        for data_name in data:
            data_path = data[data_name]["path"]
            if data_path.exists():
                df = pd.read_csv(self.met_data_path)
                # Ensure both columns are timezone-naive datetime
                if "time" in df.columns:
                    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
                if "date_creation" in df.columns:
                    df["date_creation"] = pd.to_datetime(df["date_creation"]).dt.tz_localize(None)
            else:
                df = pd.DataFrame()
            
            if data_name == "met_data":
                self.data_met = df
            elif data_name == "forecast":
                self.data_forecast = df
        

    
    def save_data_dir(self)->None:
        """
        Save different data frames to data_dir
        """
        self.data_met.to_csv(self.met_data_path, index=False)
        self.data_forecast.to_csv(self.forecast_data_path, index=False)


    def met_request_data(self, location_name:str)->pd.DataFrame:
        """
        Request data from MET office API

        Args:
            location_name (str): Name of location, list of locations in gloabls.py
        Returns:
            pd.DataFrame: df with MET office data
        """
        params = {
            "includeLocationName": "true",
            "latitude": LOCATIONS[location_name]["latitude"],
            "longitude": LOCATIONS[location_name]["longitude"],
        }
        headers = {
            "accept": "application/json",
            "apikey": self.api_key,
        }

        response = requests.get(self.url, params=params, headers=headers, verify=True)
        response.raise_for_status()
        data = response.json()
        df = self.clean_met_data(met_response_json=data, location_name=location_name)
        return df
    

    def clean_met_data(self, met_response_json:dict, location_name:str)->pd.DataFrame:
        """
        Flatten the data response from MET. This is actually just getting to the data we want

        Args:
            met_response_json (dict): JSON response from MET
            location_name (str): Name of location, list of locations in globals.py
        Returns:
            pd.DataFrame: df of forecast and actuals. Majority is forecast with one or two actuals
        """
        df = pd.DataFrame(met_response_json["features"][0]["properties"]["timeSeries"])
        df["date_creation"] = pd.Timestamp.now().tz_localize(None).floor('min')
        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
        df["forecast"] = df["time"] >= df["date_creation"]
        df["location_name"] = location_name
        return df


    def update_met_data(self, location_names:list[str]=[])->pd.DataFrame:
        """
        Get the latest forecast and actuals from MET and update our database
        Args:
            location_names (list[str]): List of location names in globals.py to update. If empty list is provided all locations are updated
        Returns:
            pd.DataFrame: df of forecast and actuals. Previous forecast is overwritten with the latest forecast
        """
        self.load_data_dir()
        if location_names==[]:
            location_names = list(LOCATIONS.keys())
        for location_name in location_names:
            try:
                df_location = self.met_request_data(location_name=location_name)
            except Exception as e:
                logger.error(f"Error loading data for {location_name} - skipping location update - error: {e}")
                continue
            # filter out previous forecast data for this location
            if len(self.data_met) > 0:
                earliest_new_time = df_location[df_location["location_name"]==location_name]["time"].min()
                filter_out_location_forecast = ~((self.data_met["location_name"]==location_name) & (self.data_met["time"] >= earliest_new_time))
                self.data_met = self.data_met[filter_out_location_forecast]
            # append new forecast data
            self.data_met = pd.concat([self.data_met, df_location], ignore_index=True)
            logger.info(f"Updated data for {location_name}")
        return self.data_met
    
    
    def update_scores(self) -> pd.DataFrame:
        """
        update the met dataframe with the wind and rain scores
        """

        if self.data_met.empty:
            raise ValueError("No MET data available. Run update_met_data before scoring.")

        def _rain_score(row: pd.Series) -> float:
            prob_rain = row.get("probOfRain")
            prob_heavy_rain = row.get("probOfHeavyRain")
            if pd.isna(prob_rain) or pd.isna(prob_heavy_rain):
                return float("nan")
            return rain_prob_score(float(prob_rain), float(prob_heavy_rain))

        def _wind_score(row: pd.Series) -> float:
            location = row.get("location_name")
            if pd.isna(location):
                return float("nan")
            try:
                vulnerable_direction = LOCATIONS[str(location)]["vulnerable_wind_direction"]
            except KeyError as exc:
                raise KeyError(f"Location '{location}' not defined in LOCATIONS.") from exc

            wind_direction = row.get("windDirectionFrom10m")
            wind_speed = row.get("windSpeed10m")
            if pd.isna(wind_direction) or pd.isna(wind_speed):
                return float("nan")

            return wind_score(
                tuple(vulnerable_direction),
                float(wind_direction),
                float(wind_speed),
            )

        self.data_met["score_rain"] = self.data_met.apply(_rain_score, axis=1)
        self.data_met["score_wind"] = self.data_met.apply(_wind_score, axis=1)
        self.data_met["score"] = (self.data_met["score_rain"] + self.data_met["score_wind"]) / 2
        logger.info("Calculated raw scores, eg. wind and rain")

        return self.data_met

    
    def update_forecast(self):
        """
        Using weather data create/update the vis forecast score and save it down in data_dir/forecast.csv
        """
        df_new_forecast = create_forecast(self.data_met)
        earliest_new_forecast = df_new_forecast["time"].min()
        if "time" in self.data_forecast.columns:
            mask_keep_old_forecast = self.data_forecast["time"] < earliest_new_forecast
            self.data_forecast = pd.concat([df_new_forecast, self.data_forecast[mask_keep_old_forecast]], ignore_index=True)
        else:
            self.data_forecast = df_new_forecast
        logger.info("Calculated vis forecast scores")
        return self.data_forecast

    
    def output_forecast_json(self):
        """
        Transform forecast.csv into a usable json format for the front end webpage. 
        Only include the morning and afternoon forecast for each day
        """
        now = pd.Timestamp.now()
        cutoff = now - pd.Timedelta(days=7)
        mask_recent = self.data_forecast["time"] > cutoff
        df_recent = self.data_forecast.loc[mask_recent].copy()
        df_recent["date"] = df_recent["time"].dt.date
        df_recent["period"] = df_recent["time"].dt.hour.apply(lambda hr: "morning" if hr < 12 else "afternoon")
        grouped = (
            df_recent.groupby(["location_name", "date", "period"], as_index=False)[["vis_score", "score_wind", "score_rain"]]
            .mean()
        )
        rag_thresholds = sorted(VIS_FORECAST_RAG_SCORE.items())
        def _rag_from_score(score: float) -> str:
            for threshold, rag in rag_thresholds:
                if score <= threshold:
                    return rag
            return rag_thresholds[-1][1]
        grouped = grouped.sort_values(["date", "location_name", "period"])
        forecasts = []
        for _, row in grouped.iterrows():
            location_name = row["location_name"]
            location_meta = LOCATIONS.get(location_name, {})
            latitude = location_meta.get("latitude")
            longitude = location_meta.get("longitude")
            score = float(row["vis_score"])
            wind_score = float(row["score_wind"])
            rain_score = float(row["score_rain"])
            forecasts.append(
                {
                    "date": row["date"].isoformat(),
                    "period": row["period"],
                    "location_name": location_name,
                    "latitude": latitude,
                    "longitude": longitude,
                    "vis_score": score,
                    "vis_rag": _rag_from_score(score),
                    "score_wind": wind_score,
                    "score_rain": rain_score,
                }
            )
        self.json_data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.json_data_path, "w", encoding="utf-8") as json_file:
            json.dump(forecasts, json_file, indent=2)
        logger.info(f"Saved {self.json_data_path}")
    
    
    def update(self):
        """
        Complete a full update - met download, score calculation, vis forecast calculation, save dataframes
        """
        # self.update_met_data()
        self.update_scores()
        self.update_forecast()
        self.save_data_dir()
        self.output_forecast_json()
        