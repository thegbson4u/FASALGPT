import requests
import os

def get_weather(city):
    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        raise ValueError("OPENWEATHER_API_KEY not found")

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }

    response = requests.get(url, params=params)

    if response.status_code == 401:
        raise ValueError("Invalid or inactive OpenWeather API key")

    response.raise_for_status()
    data = response.json()

    return {
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "rainfall": data.get("rain", {}).get("1h", 0),
        "wind_speed": data["wind"]["speed"],
        "condition": data["weather"][0]["description"]
    }
