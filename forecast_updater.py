from vis_forecast.data_loader import MetOfficeDataLoader
import sys
from dotenv import load_dotenv
from loguru import logger
import time
import argparse
import requests

from urllib3.exceptions import InsecureRequestWarning
# Suppress the warnings from urllib3
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

def get_args():
    parser = argparse.ArgumentParser(description="Run the vis_forecast MET scraper")
    parser.add_argument('--scrape_freq', type=int, help='Frequency in seconds to run the scraper, default=10500', default=10500)
    args = parser.parse_args()
    return args

def sleep(seconds:int):
    logger.info(f"Next update in {seconds/60:.0f} minutes")
    time.sleep(seconds)
    


def main():
    if not load_dotenv():
        logger.error("Failed to load .env file")
        sys.exit()
    args = get_args()
    logger.info(f"Starting vis_forecast MET scraper - download forecast update every {args.scrape_freq/60:0.0f} minutes")

    metoffice = MetOfficeDataLoader()
    while True:
        try:
            metoffice.update_met_data()
            logger.info(f"Finished updating MET data")
        except Exception as e:
            logger.error(f"Error updating MET data - error: {e}")
        sleep(args.scrape_freq)



if __name__ == "__main__":
    logger.add("forecast_updater.log")
    main()
