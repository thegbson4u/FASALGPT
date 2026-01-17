import os
import requests


def get_weather(city: str):
    """
    Fetch current weather data from OpenWeather API.
    Works on Streamlit Cloud using Secrets.
    """

    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENWEATHER_API_KEY not found. "
            "Add it in Streamlit Secrets."
        )

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }

    response = requests.get(url, params=params, timeout=10)

    # Handle common API errors clearly
    if response.status_code == 401:
        raise RuntimeError("❌ Invalid OpenWeather API key")
    if response.status_code == 404:
        raise RuntimeError("❌ City not found")

    response.raise_for_status()
    data = response.json()

    # Safe rainfall handling
    rainfall = 0.0
    if "rain" in data:
        rainfall = data["rain"].get("1h", 0.0) or data["rain"].get("3h", 0.0)

    return {
        "temperature": round(data["main"]["temp"], 1),
        "humidity": data["main"]["humidity"],
        "rainfall": rainfall,
        "wind_speed": data["wind"]["speed"],
        "condition": data["weather"][0]["description"].title(),
        "location": f"{data['name']}, {data['sys']['country']}"
    }
