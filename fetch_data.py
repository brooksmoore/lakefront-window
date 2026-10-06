"""One-time data pull for Lakefront Window.

Downloads everything run.py needs into ./data and ./models so that
`python run.py` works offline afterwards:

  1. Chicago Park District beach weather sensors (Chicago Data Portal k7hf-8y75),
     Oak Street Weather Station, all hourly rows.
  2. Chicago Midway (MDW) hourly airport observations from the Iowa Environmental
     Mesonet ASOS archive (NOAA/FAA ASOS data), same period.
  3. The current NWS hourly forecast for the Midway gridpoint (api.weather.gov).
  4. TabPFN-v2 classifier weights from Hugging Face (Prior-Labs/TabPFN-v2-clf).

No API keys or accounts are needed for any of these.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
UA = {"User-Agent": "lakefront-window/0.1 (open-source hobby project)"}

BEACH_URL = "https://data.cityofchicago.org/resource/k7hf-8y75.csv"
BEACH_STATION = "Oak Street Weather Station"
MDW_LATLON = (41.786, -87.752)  # Chicago Midway airport


def get(url: str, params=None, tries: int = 4, **kw) -> requests.Response:
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=120, **kw)
            if r.status_code == 200:
                return r
            print(f"  HTTP {r.status_code} for {url}; retrying", file=sys.stderr)
        except requests.RequestException as e:  # pragma: no cover - network
            print(f"  {e}; retrying", file=sys.stderr)
        time.sleep(2 * (i + 1))
    raise SystemExit(f"Failed to fetch {url}")


def fetch_beach() -> None:
    cols = ("measurement_timestamp,air_temperature,wet_bulb_temperature,humidity,"
            "rain_intensity,interval_rain,precipitation_type,wind_direction,wind_speed,"
            "maximum_wind_speed,barometric_pressure,solar_radiation")
    out = DATA / "beach_oak_street.csv"
    chunks, offset, limit = [], 0, 50000
    while True:
        r = get(BEACH_URL, params={
            "$select": cols,
            "$where": f"station_name='{BEACH_STATION}'",
            "$order": "measurement_timestamp",
            "$limit": limit,
            "$offset": offset,
        })
        lines = r.text.splitlines()
        header, rows = lines[0], lines[1:]
        if not chunks:
            chunks.append(header)
        chunks.extend(rows)
        print(f"  beach rows so far: {len(chunks) - 1}")
        if len(rows) < limit:
            break
        offset += limit
    out.write_text("\n".join(chunks) + "\n")
    print(f"wrote {out} ({len(chunks) - 1} rows)")


def fetch_mdw() -> None:
    out = DATA / "mdw_asos.csv"
    today = dt.date.today() + dt.timedelta(days=1)
    params = [("station", "MDW")]
    params += [("data", d) for d in ("tmpf", "sknt", "drct", "relh", "p01i")]
    params += [("year1", 2015), ("month1", 5), ("day1", 1),
               ("year2", today.year), ("month2", today.month), ("day2", today.day),
               ("tz", "America/Chicago"), ("format", "onlycomma"), ("latlon", "no"),
               ("missing", "M"), ("trace", "T"), ("report_type", 3)]
    r = get("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py", params=params)
    out.write_text(r.text)
    print(f"wrote {out} ({r.text.count(chr(10)) - 1} rows)")


def fetch_nws() -> None:
    lat, lon = MDW_LATLON
    p = get(f"https://api.weather.gov/points/{lat},{lon}").json()["properties"]
    fc = get(p["forecastHourly"]).json()
    fc["_fetched_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    fc["_gridpoint"] = f"{p['gridId']}/{p['gridX']},{p['gridY']}"
    out = DATA / "nws_hourly_mdw.json"
    out.write_text(json.dumps(fc, indent=1))
    n = len(fc["properties"]["periods"])
    print(f"wrote {out} ({n} hourly periods, gridpoint {fc['_gridpoint']}, "
          f"generated {fc['properties'].get('generatedAt')})")


def fetch_model() -> None:
    os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(MODELS))
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cpu")
    # Fitting a tiny dummy problem forces the checkpoint download into ./models.
    import numpy as np
    clf.fit(np.random.rand(20, 2), np.array([0, 1] * 10))
    ckpts = sorted(MODELS.glob("tabpfn-v2-classifier*.ckpt"))
    print(f"model weights: {[c.name for c in ckpts]}")


def main() -> None:
    DATA.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    what = set(sys.argv[1:]) or {"beach", "mdw", "nws", "model"}
    if "beach" in what:
        fetch_beach()
    if "mdw" in what:
        fetch_mdw()
    if "nws" in what:
        fetch_nws()
    if "model" in what:
        fetch_model()


if __name__ == "__main__":
    main()
