"""Data preparation and the 'good-outside-hour' label for Lakefront Window."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# ---------------------------------------------------------------------------
# The target. Deliberately simple and transparent: an hour at the Oak Street
# lakefront sensor counts as a "good outside hour" when ALL of these hold.
# ---------------------------------------------------------------------------
TEMP_MIN_C = 10.0   # 50 F
TEMP_MAX_C = 27.0   # ~80 F
WIND_MAX_MS = 9.0   # ~20 mph; lakefront: sensor's hourly max gust, regional: sustained wind
HOURS = list(range(7, 20))  # 7:00 .. 19:00 local; daylight-ish hours only

FEATURES = [
    "hour", "doy_sin", "doy_cos",
    "temp_c", "wind_ms", "wdir_sin", "wdir_cos", "rh", "precip",
]

COMPASS = {d: i * 22.5 for i, d in enumerate(
    "N NNE NE ENE E ESE SE SSE S SSW SW WSW W WNW NW NNW".split())}


def good_hour(temp_c, wind_ms, wet) -> np.ndarray:
    """Apply the label thresholds (works on lakefront obs OR on regional inputs)."""
    temp_c = np.asarray(temp_c, float)
    wind_ms = np.asarray(wind_ms, float)
    wet = np.asarray(wet, bool)
    return ((temp_c >= TEMP_MIN_C) & (temp_c <= TEMP_MAX_C)
            & (wind_ms < WIND_MAX_MS) & ~wet).astype(int)


def load_beach() -> pd.DataFrame:
    b = pd.read_csv(DATA / "beach_oak_street.csv", parse_dates=["measurement_timestamp"])
    b = b[b.measurement_timestamp.dt.minute == 0].copy()
    b["time"] = b.measurement_timestamp
    # Basic QC: drop sensor sentinels / impossible values.
    b.loc[(b.wind_speed < 0) | (b.wind_speed > 40), "wind_speed"] = np.nan
    b.loc[(b.maximum_wind_speed < 0) | (b.maximum_wind_speed > 60), "maximum_wind_speed"] = np.nan
    b.loc[(b.air_temperature < -35) | (b.air_temperature > 45), "air_temperature"] = np.nan
    b["wet"] = (b.interval_rain.fillna(0) > 0) | (b.rain_intensity.fillna(0) > 0) | \
               b.precipitation_type.fillna(0).isin([60, 70])
    b = b.dropna(subset=["air_temperature", "maximum_wind_speed"])
    b = b.drop_duplicates("time").set_index("time")
    out = pd.DataFrame({
        "lake_temp_c": b.air_temperature,
        "lake_wind_ms": b.wind_speed,
        "lake_gust_ms": b.maximum_wind_speed,
        "lake_wet": b.wet.astype(int),
    })
    out["label"] = good_hour(out.lake_temp_c, out.lake_gust_ms, out.lake_wet)
    return out


def load_mdw() -> pd.DataFrame:
    m = pd.read_csv(DATA / "mdw_asos.csv", na_values=["M"], dtype={"p01i": str})
    m["valid"] = pd.to_datetime(m["valid"])
    # Routine METARs are issued at hh:53; assign each to the following top of hour.
    m["time"] = m["valid"].dt.ceil("h")
    m = m.drop_duplicates("time", keep="last").set_index("time")
    precip = m.p01i.replace({"T": "0.0001"}).astype(float)
    out = pd.DataFrame({
        "temp_c": (m.tmpf - 32) * 5 / 9,
        "wind_ms": m.sknt * 0.514444,
        "wdir": m.drct,
        "rh": m.relh,
        "precip": (precip > 0).astype(float).where(precip.notna()),
    })
    return out.dropna()


def add_time_and_wind_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idx = df.index
    df["hour"] = idx.hour
    doy = idx.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    calm = (df.wind_ms < 0.3) | (df.wdir == 0)
    rad = np.deg2rad(df.wdir)
    df["wdir_sin"] = np.where(calm, 0.0, np.sin(rad))
    df["wdir_cos"] = np.where(calm, 0.0, np.cos(rad))
    return df


def build_dataset() -> pd.DataFrame:
    """Hourly table: Midway inputs (stand-in for a regional forecast) + lakefront label."""
    df = load_mdw().join(load_beach(), how="inner")
    df = df[df.index.hour.isin(HOURS)]
    return add_time_and_wind_features(df)


def load_forecast() -> tuple[pd.DataFrame, dict]:
    """NWS hourly forecast for the Midway gridpoint, converted to model inputs."""
    fc = json.loads((DATA / "nws_hourly_mdw.json").read_text())
    rows = []
    for p in fc["properties"]["periods"]:
        t = pd.Timestamp(p["startTime"]).tz_convert("America/Chicago").tz_localize(None)
        temp = p["temperature"]
        temp_c = (temp - 32) * 5 / 9 if p.get("temperatureUnit", "F") == "F" else temp
        ws = [float(x) for x in str(p["windSpeed"]).replace("mph", "").split("to")]
        rows.append({
            "time": t,
            "temp_c": temp_c,
            "wind_ms": float(np.mean(ws)) * 0.44704,
            "wdir": COMPASS.get(p.get("windDirection") or "", 0.0),
            "rh": (p.get("relativeHumidity") or {}).get("value") or np.nan,
            "pop": ((p.get("probabilityOfPrecipitation") or {}).get("value") or 0) / 100.0,
            "short": p.get("shortForecast", ""),
            "is_day": p.get("isDaytime", True),
        })
    f = pd.DataFrame(rows).set_index("time")
    f["rh"] = f.rh.fillna(f.rh.median() if f.rh.notna().any() else 70.0)
    meta = {
        "generated_at": fc["properties"].get("generatedAt"),
        "fetched_at": fc.get("_fetched_at"),
        "gridpoint": fc.get("_gridpoint"),
    }
    return add_time_and_wind_features(f), meta
