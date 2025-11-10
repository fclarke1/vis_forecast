# Sea Visibility Forecast

Using data from the Met Office and manually selected locations create a simple visibility forecast and visualise on an interactive map

## Installation

1. On MET office apply to get your free api key for global spot weather forecasts and input into a .env file:
```
MET_OFFICE_API_KEY=your_api_key
```

2. Create a .venv, eg with uv:
```
uv sync
uv venv
```

## Execution

To gather data and calculate forecast:

`python forecast_updater.py --scrape_freq=0`

`scrape_freq` is the number of seconds between each MET download, if 0 then it's only run once