"""
Weather tool -- live external data via Open-Meteo (no API key required).

Two calls: geocode the place name to coordinates, then fetch the forecast.
Geocoding results are cached in memory because people ask about the same two
or three places all day and a network round-trip is pure latency.

The tool returns a small, flat, already-interpreted structure. The raw API
response is deliberately not passed to the model -- it is 200 lines of nested
JSON that would cost tokens, slow the response, and give the model more
opportunity to misread a field.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from .base import Tool, ToolResult

logger = logging.getLogger(__name__)

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather interpretation codes -> plain English fit for speech.
WMO = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "foggy", 51: "light drizzle", 53: "drizzle",
    55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "rain showers", 81: "rain showers", 82: "violent rain showers",
    85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorms", 96: "thunderstorms with hail",
    99: "thunderstorms with hail",
}


class WeatherTool(Tool):
    name = "get_weather"
    description = (
        "Get current conditions or a daily forecast for a place. Use this for any "
        "question about weather, temperature, rain, or whether to carry an umbrella. "
        "Supports today, tomorrow, or a day up to 6 days ahead."
    )

    def __init__(self, config):
        self.cfg = config
        self._geo_cache: dict[str, tuple[float, float, str]] = {}

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City or place name, e.g. 'Hyderabad'. "
                                   "Omit to use the user's default location.",
                },
                "day_offset": {
                    "type": "integer",
                    "description": "0 for today, 1 for tomorrow, up to 6.",
                    "minimum": 0,
                    "maximum": 6,
                },
            },
            "required": [],
        }

    # -- helpers -----------------------------------------------------------
    def _client(self):
        import httpx
        return httpx.Client(timeout=self.cfg.http_timeout_s)

    def _geocode(self, place: str):
        key = place.lower().strip()
        if key in self._geo_cache:
            return self._geo_cache[key]
        with self._client() as client:
            resp = client.get(GEOCODE_URL, params={"name": place, "count": 1, "language": "en"})
            resp.raise_for_status()
            results = (resp.json() or {}).get("results") or []
        if not results:
            return None
        top = results[0]
        label = top.get("name", place)
        if top.get("country"):
            label = f"{label}, {top['country']}"
        found = (float(top["latitude"]), float(top["longitude"]), label)
        self._geo_cache[key] = found
        return found

    # -- execution ---------------------------------------------------------
    def execute(self, location: str = "", day_offset: int = 0) -> ToolResult:
        place = (location or self.cfg.default_location).strip()
        try:
            day_offset = max(0, min(6, int(day_offset)))
        except (TypeError, ValueError):
            day_offset = 0

        try:
            geo = self._geocode(place)
        except Exception as exc:
            return self._fail(f"Could not look up '{place}': {exc}")
        if geo is None:
            return self._fail(f"I could not find a place called '{place}'.")
        lat, lon, label = geo

        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
                     "precipitation_probability_max",
            "timezone": "auto",
            "forecast_days": 7,
        }
        if day_offset == 0:
            params["current"] = "temperature_2m,apparent_temperature,weather_code,relative_humidity_2m"

        try:
            with self._client() as client:
                resp = client.get(FORECAST_URL, params=params)
                resp.raise_for_status()
                payload = resp.json()
        except Exception as exc:
            return self._fail(f"The weather service did not respond: {exc}")

        try:
            daily = payload["daily"]
            idx = day_offset
            code = int(daily["weather_code"][idx])
            result = {
                "location": label,
                "day": ["today", "tomorrow"][idx] if idx < 2
                       else (date.today() + timedelta(days=idx)).strftime("%A"),
                "condition": WMO.get(code, "mixed"),
                "high_c": round(float(daily["temperature_2m_max"][idx])),
                "low_c": round(float(daily["temperature_2m_min"][idx])),
                "rain_probability_pct": int(daily["precipitation_probability_max"][idx] or 0),
            }
            if day_offset == 0 and "current" in payload:
                cur = payload["current"]
                result["current_temp_c"] = round(float(cur["temperature_2m"]))
                result["feels_like_c"] = round(float(cur["apparent_temperature"]))
                result["current_condition"] = WMO.get(int(cur["weather_code"]), "mixed")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            return self._fail(f"The weather data was in an unexpected shape: {exc}")

        return self._ok(result, location=label, day_offset=day_offset)
