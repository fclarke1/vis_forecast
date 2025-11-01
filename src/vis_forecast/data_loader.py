import requests
from pathlib import Path
import pandas as pd
import os
from loguru import logger

from vis_forecast.globals import LOCATIONS


class MetOfficeDataLoader:
    def __init__(self, api_key:str=None, data_dir:str="./data"):
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.environ["MET_OFFICE_API_KEY"]
        self.url = "https://data.hub.api.metoffice.gov.uk/sitespecific/v0/point/three-hourly"
        self.data_dir = Path(data_dir)
        self.load_data_dir()
    

    def load_data_dir(self) -> None:
        """
        Load previously recorded data from MET office. If no data is there then create fresh csv files
        """
        self.met_data_path = self.data_dir / "met_data.csv"
        if self.met_data_path.exists():
            self.data_met = pd.read_csv(self.met_data_path)
            # Ensure both columns are timezone-naive datetime
            if "time" in self.data_met.columns:
                self.data_met["time"] = pd.to_datetime(self.data_met["time"]).dt.tz_localize(None)
            if "date_creation" in self.data_met.columns:
                self.data_met["date_creation"] = pd.to_datetime(self.data_met["date_creation"]).dt.tz_localize(None)
        else:
            self.data_met = pd.DataFrame()

    
    def save_data_dir(self)->None:
        """
        Save different data frames to data_dir
        """
        self.data_met.to_csv(self.met_data_path, index=False)


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
            print(location_name)
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
        self.save_data_dir()
        return self.data_met
        
