import asyncio
import copy
import logging
import math
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import httpx
from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from pydantic import BaseModel, Field

from models import MODEL_IDS

# Load GEMINI_API_KEY and GOOGLE_WEATHER_API_KEY from local .env at startup
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rainready")

# Official google-genai (>=3.2.0) clients:
# - Stable v1 client for Gemini 4 Interactions API
# - v1beta client for Gemini Decision 1 (Decisions API)
client: Optional[genai.Client] = None
beta_client: Optional[genai.Client] = None


def _init_genai_clients() -> tuple[genai.Client, genai.Client]:
    global client, beta_client
    load_dotenv(override=True)
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY is not configured in .env or environment variables.",
        )
    if client is None or beta_client is None:
        client = genai.Client()
        beta_client = genai.Client(http_options={"api_version": "v1beta"})
    return client, beta_client


if os.environ.get("GEMINI_API_KEY", "").strip():
    client = genai.Client()
    beta_client = genai.Client(http_options={"api_version": "v1beta"})

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

CLOUD_RUN_PROJECT = "geminisprints26blr-3834"
GOOGLE_FORECAST_HOURS_URL = "https://weather.googleapis.com/v1/forecast/hours:lookup"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
SLOT_LABELS = ["Now", "+1h", "+2h", "+3h", "+4h", "+5h"]

SYSTEM_INSTRUCTION = """You are RainReady's Hyperlocal Rainfall Intensity + Duration Flood Engine.
Synthesize the 6-hour hourly rainfall forecast, pre-computed continuous rain duration & drainage threshold metrics, proximity to known historical flood-prone hotspots, neighborhood terrain metadata, timestamped citizen water reports, and calibrated Gemini Decision 1 probabilities/scores:
1. Populate `expected_rainfall_summary` with a crisp human-readable total (e.g., "135 mm expected over the next 3 hours" during heavy bursts, or "0.8 mm expected over the next 6 hours" when light).
2. Populate `continuous_rain_hours` (sustained hours > 2.0 mm/hr) and `hours_until_flood_risk` (float hours until street flooding begins, or null when safe).
3. Populate `duration_alert_headline`:
   - When the location is at WATCH or HIGH flood risk (or rainfall exceeds drainage thresholds / citizen waterlogging is active), use this exact warning format when rain-driven:
     "Flood risk in [Location] if rain continues at [X] mm/hr for [Y] hours."
     (If citizen reports already confirm active street waterlogging despite light forecast rain, clearly state active waterlogging is reported in [Location] and caution commuters immediately.)
   - When the location is SAFE / LOW risk, write a clear, reassuring all-clear headline for [Location] (e.g., "No flood risk expected in [Location] over the next 6 hours.").
4. Populate `hourly_breakdown` with all 6 forecast slots ("Now", "+1h", "+2h", "+3h", "+4h", "+5h") matching the provided hourly forecast rates and `exceeds_drain_capacity` flags.
5. List exact causal drivers in `visible_signals`, write a concise verification note in `responder_summary`, and provide 2 to 3 plain-language `resident_actions` tailored to [Location].
"""


# ---------------------------------------------------------------------------
# Native Structured Output Schema & API Models
# ---------------------------------------------------------------------------
class HourlyBreakdownItem(BaseModel):
    hour_label: str = Field(
        ...,
        description="Forecast slot label ('Now', '+1h', '+2h', '+3h', '+4h', '+5h').",
    )
    mm_per_hr: float = Field(
        ...,
        ge=0.0,
        description="Forecast rainfall intensity in mm/hr for this hour.",
    )
    exceeds_drain_capacity: bool = Field(
        ...,
        description="True when mm_per_hr >= the location's drainage_threshold_mm_per_hr.",
    )


class Stage2ActionableReasoning(BaseModel):
    visible_signals: list[str] = Field(
        ...,
        min_length=2,
        max_length=6,
        description="Exact observable flood risk signal drivers (rainfall rate, duration, historical flood hotspot proximity, resident reports).",
    )
    responder_summary: str = Field(
        ...,
        description="Concise verification note for human emergency reviewers.",
    )
    resident_actions: list[str] = Field(
        ...,
        min_length=2,
        max_length=3,
        description="2 to 3 clear, practical next steps in plain language for residents.",
    )
    expected_rainfall_summary: str = Field(
        ...,
        description="Concise expected rainfall summary, e.g., '135 mm expected over the next 3 hours'.",
    )
    continuous_rain_hours: float = Field(
        ...,
        ge=0.0,
        description="Sustained consecutive hours of rain (> 2.0 mm/hr).",
    )
    hours_until_flood_risk: Optional[float] = Field(
        default=None,
        description="Estimated hours until street waterlogging/flooding occurs (e.g., 1.5 or 2.0), or null if safe.",
    )
    duration_alert_headline: str = Field(
        ...,
        description=(
            "When at risk: 'Flood risk in [Location] if rain continues at [X] mm/hr for [Y] hours.' "
            "When safe: clear all-clear message for [Location]."
        ),
    )
    hourly_breakdown: list[HourlyBreakdownItem] = Field(
        ...,
        min_length=6,
        max_length=6,
        description="6-hour forecast breakdown (Now through +5h) for the UI intensity strip.",
    )


class AnalyzeRequest(BaseModel):
    simulated_rain_mm: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=500.0,
        description="Optional simulated rainfall intensity override (mm/hr) from the UI.",
    )


class AnalyzeLocationRequest(BaseModel):
    lat: float = Field(..., ge=-90.0, le=90.0, description="Latitude of GPS or clicked spot.")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Longitude of GPS or clicked spot.")
    simulated_rain_mm: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=500.0,
        description="Optional simulated 3-hour storm intensity override (mm/hr).",
    )
    is_gps: bool = Field(
        default=False,
        description="True when triggered from the user's live GPS location.",
    )


class ReportRequest(BaseModel):
    zone_id: Optional[str] = Field(
        default=None,
        description="Optional target hotspot zone ID or dynamic spot ID.",
    )
    lat: Optional[float] = Field(
        default=None,
        ge=-90.0,
        le=90.0,
        description="Exact latitude for the live waterlogging map pin.",
    )
    lon: Optional[float] = Field(
        default=None,
        ge=-180.0,
        le=180.0,
        description="Exact longitude for the live waterlogging map pin.",
    )
    place_name: Optional[str] = Field(
        default=None,
        max_length=160,
        description="Optional human-readable place name for the report pin.",
    )
    depth: str = Field(
        ...,
        min_length=2,
        max_length=80,
        description="Reported waterlogging depth or condition (Ankle-Deep, Knee-Deep, Blocked Drain).",
    )
    description: str = Field(
        default="",
        max_length=500,
        description="Additional landmark or street context from the resident.",
    )
    simulated_rain_mm: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=500.0,
        description="Optional active heavy-rain override if enabled in UI.",
    )


# ---------------------------------------------------------------------------
# 8 Historical Flood-Prone Hotspots (4 Kochi + 4 Bengaluru)
# ---------------------------------------------------------------------------
def _make_zone(
    zone_id: str,
    city: str,
    name: str,
    subtitle: str,
    lat: float,
    lng: float,
    elevation_m: float,
    threshold_mm: float,
    sat_hours: float,
    impervious_pct: int,
    canal_basin: str,
    historical_flood_note: str,
    coords: list[list[float]],
) -> dict[str, Any]:
    return {
        "id": zone_id,
        "city": city,
        "is_custom_location": False,
        "name": name,
        "subtitle": subtitle,
        "lat": lat,
        "lng": lng,
        "historical_flood_note": historical_flood_note,
        "metadata": {
            "elevation_m": elevation_m,
            "drainage_capacity_mm_hr": threshold_mm,
            "drainage_threshold_mm_per_hr": threshold_mm,
            "saturation_hours": sat_hours,
            "impervious_surface_pct": impervious_pct,
            "canal_basin": canal_basin,
            "drainage_profile": historical_flood_note,
        },
        "polygon": {
            "type": "Feature",
            "properties": {"zone_id": zone_id, "city": city},
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords],
            },
        },
        "reports": [],
        "weather": None,
        "latest_evaluation": None,
        "decisions_telemetry": None,
        "routing_info": None,
        "last_interaction_id": None,
        "review_status": "UNASSESSED",
        "approved": False,
        "evaluated_at": None,
        "approved_alert": None,
    }


def _build_initial_zones() -> dict[str, dict[str, Any]]:
    return {
        # --- KOCHI HISTORICAL FLOOD HOTSPOTS ---
        "south_railway_underpass": _make_zone(
            zone_id="south_railway_underpass",
            city="Kochi",
            name="South Railway Underpass",
            subtitle="Kochi · Ernakulam Junction South · Karshaka Rd",
            lat=9.9696,
            lng=76.2908,
            elevation_m=1.8,
            threshold_mm=25.0,
            sat_hours=1.5,
            impervious_pct=89,
            canal_basin="Perandoor Canal (180m east)",
            historical_flood_note=(
                "Low-lying railway box-culvert underpass (1.8m MSL); historically floods within "
                "90 mins of >=25 mm/hr rain or Perandoor Canal tidal surcharge."
            ),
            coords=[
                [76.2878, 9.9672],
                [76.2938, 9.9672],
                [76.2942, 9.9718],
                [76.2882, 9.9721],
                [76.2878, 9.9672],
            ],
        ),
        "panampilly_nagar": _make_zone(
            zone_id="panampilly_nagar",
            city="Kochi",
            name="Panampilly Nagar / TP Canal",
            subtitle="Kochi · Main Avenue & Thevara–Perandoor Canal Grid",
            lat=9.9605,
            lng=76.2975,
            elevation_m=2.4,
            threshold_mm=30.0,
            sat_hours=2.5,
            impervious_pct=78,
            canal_basin="Thevara–Perandoor Canal (adjacent)",
            historical_flood_note=(
                "Reclaimed low-lying marshland grid bordering Thevara–Perandoor Canal; historically "
                "waterlogs internal cross-roads when >=30 mm/hr rain persists for 2.5 hours."
            ),
            coords=[
                [76.2936, 9.9554],
                [76.3022, 9.9554],
                [76.3025, 9.9652],
                [76.2940, 9.9652],
                [76.2936, 9.9554],
            ],
        ),
        "mg_road_padma": _make_zone(
            zone_id="mg_road_padma",
            city="Kochi",
            name="MG Road / Padma Junction",
            subtitle="Kochi · Central Commercial Spine · Mullassery Basin",
            lat=9.9806,
            lng=76.2836,
            elevation_m=2.1,
            threshold_mm=35.0,
            sat_hours=2.0,
            impervious_pct=94,
            canal_basin="Mullassery Canal (250m south-west)",
            historical_flood_note=(
                "94% impervious commercial corridor draining to Mullassery Canal; silted box drains "
                "historically pool curb-to-shopfront within 2 hours of >=35 mm/hr rain."
            ),
            coords=[
                [76.2798, 9.9762],
                [76.2874, 9.9762],
                [76.2878, 9.9848],
                [76.2801, 9.9850],
                [76.2798, 9.9762],
            ],
        ),
        "edappally_toll": _make_zone(
            zone_id="edappally_toll",
            city="Kochi",
            name="Edappally Toll Junction",
            subtitle="Kochi · NH-66 / NH-544 Convergence · Edappally Thodu",
            lat=10.0261,
            lng=76.3083,
            elevation_m=3.6,
            threshold_mm=30.0,
            sat_hours=2.0,
            impervious_pct=91,
            canal_basin="Edappally Thodu (320m west)",
            historical_flood_note=(
                "Major highway & flyover ramp catchment funneling sheet runoff toward Edappally Thodu; "
                "service lanes historically flood after 2 hours of >=30 mm/hr rain."
            ),
            coords=[
                [76.3038, 10.0218],
                [76.3128, 10.0218],
                [76.3131, 10.0304],
                [76.3041, 10.0306],
                [76.3038, 10.0218],
            ],
        ),
        # --- BENGALURU HISTORICAL FLOOD HOTSPOTS ---
        "silk_board_junction": _make_zone(
            zone_id="silk_board_junction",
            city="Bengaluru",
            name="Silk Board Junction",
            subtitle="Bengaluru · Hosur Rd / Outer Ring Rd Valley Depression",
            lat=12.9177,
            lng=77.6233,
            elevation_m=886.0,
            threshold_mm=28.0,
            sat_hours=1.5,
            impervious_pct=93,
            canal_basin="Madiwala Lake Stormwater Valley",
            historical_flood_note=(
                "Low-point arterial interchange in the Roopena Agrahara–Madiwala valley; service lanes "
                "and flyover ramps historically flood within 90 mins of >=28 mm/hr rain."
            ),
            coords=[
                [77.6192, 12.9145],
                [77.6275, 12.9145],
                [77.6278, 12.9212],
                [77.6195, 12.9212],
                [77.6192, 12.9145],
            ],
        ),
        "bellandur_orr": _make_zone(
            zone_id="bellandur_orr",
            city="Bengaluru",
            name="Bellandur / Outer Ring Road",
            subtitle="Bengaluru · EcoSpace Tech Corridor · Bellandur Rajakaluve",
            lat=12.9304,
            lng=77.6784,
            elevation_m=879.0,
            threshold_mm=25.0,
            sat_hours=1.5,
            impervious_pct=90,
            canal_basin="Bellandur–Varthur Lake Rajakaluve",
            historical_flood_note=(
                "Reclaimed wetland tech corridor astride the primary Bellandur Lake rajakaluve outfall; "
                "historically inundates ORR service roads within 90 mins of >=25 mm/hr downpours."
            ),
            coords=[
                [77.6732, 12.9262],
                [77.6836, 12.9262],
                [77.6840, 12.9348],
                [77.6736, 12.9348],
                [77.6732, 12.9262],
            ],
        ),
        "koramangala_4th_block": _make_zone(
            zone_id="koramangala_4th_block",
            city="Bengaluru",
            name="Koramangala 4th Block",
            subtitle="Bengaluru · 80 Feet Rd & ST Bed Stormwater Valley",
            lat=12.9344,
            lng=77.6266,
            elevation_m=883.0,
            threshold_mm=30.0,
            sat_hours=2.0,
            impervious_pct=86,
            canal_basin="K-100 Citizens Waterway / Bellandur Feeder Drain",
            historical_flood_note=(
                "Low-lying residential basin bordering ST Bed and primary rajakaluve feeder; "
                "historically experiences street waterlogging after 2 hours of >=30 mm/hr rain."
            ),
            coords=[
                [77.6222, 12.9308],
                [77.6310, 12.9308],
                [77.6313, 12.9382],
                [77.6225, 12.9382],
                [77.6222, 12.9308],
            ],
        ),
        "manyata_hebbal": _make_zone(
            zone_id="manyata_hebbal",
            city="Bengaluru",
            name="Manyata Tech Park / Hebbal",
            subtitle="Bengaluru · Nagawara Lake & Hebbal Ring Rd Catchment",
            lat=13.0452,
            lng=77.6208,
            elevation_m=892.0,
            threshold_mm=30.0,
            sat_hours=2.0,
            impervious_pct=88,
            canal_basin="Hebbal–Nagawara Lake Chain",
            historical_flood_note=(
                "Depressed Ring Road & tech-campus drainage apron upstream of Nagawara Lake; "
                "historically pools rapidly when >=30 mm/hr cloudbursts exceed culvert capacity."
            ),
            coords=[
                [77.6158, 13.0412],
                [77.6258, 13.0412],
                [77.6262, 13.0494],
                [77.6162, 13.0494],
                [77.6158, 13.0412],
            ],
        ),
    }


ZONES: dict[str, dict[str, Any]] = _build_initial_zones()
DYNAMIC_LOCATIONS: dict[str, dict[str, Any]] = {}
ALL_REPORTS: list[dict[str, Any]] = []
GEOCODE_CACHE: dict[tuple[float, float], dict[str, str]] = {}

ZONE_LOCKS: dict[str, asyncio.Lock] = {zid: asyncio.Lock() for zid in ZONES}
GLOBAL_LOC_LOCK = asyncio.Lock()

_http_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    global _http_client
    _http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(10.0, connect=5.0),
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
    )
    try:
        yield
    finally:
        await _http_client.aclose()
        _http_client = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _find_nearest_hotspot(lat: float, lon: float) -> tuple[dict[str, Any], float]:
    best_zone: Optional[dict[str, Any]] = None
    best_dist = float("inf")
    for z in ZONES.values():
        d = _haversine_km(lat, lon, z["lat"], z["lng"])
        if d < best_dist:
            best_dist = d
            best_zone = z
    assert best_zone is not None
    return best_zone, round(best_dist, 2)


def _reports_near_point(lat: float, lon: float, radius_km: float = 2.0) -> list[dict[str, Any]]:
    return [
        r
        for r in ALL_REPORTS
        if _haversine_km(lat, lon, float(r["lat"]), float(r["lon"])) <= radius_km
    ]


def _format_hour_clock(hour_24: Optional[int]) -> str:
    if hour_24 is None:
        return ""
    h = int(hour_24) % 24
    suffix = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12} {suffix}"


async def _http_get_json(
    url: str, params: dict[str, Any], headers: Optional[dict[str, str]] = None
) -> dict[str, Any]:
    if _http_client is not None:
        resp = await _http_client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()
    async with httpx.AsyncClient(timeout=10.0) as fallback_client:
        resp = await fallback_client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()


async def _reverse_geocode(lat: float, lon: float) -> dict[str, str]:
    """Reverse-geocode {lat, lon} using OpenStreetMap Nominatim (`User-Agent: RainReady-App`)."""
    cache_key = (round(lat, 3), round(lon, 3))
    if cache_key in GEOCODE_CACHE:
        return GEOCODE_CACHE[cache_key]

    nearest, dist_km = _find_nearest_hotspot(lat, lon)
    fallback_city = nearest["city"] if dist_km < 30.0 else "Local Area"
    fallback_name = (
        f"Near {nearest['name']}"
        if dist_km <= 1.2
        else f"Location ({lat:.4f}°N, {lon:.4f}°E)"
    )

    try:
        data = await _http_get_json(
            NOMINATIM_REVERSE_URL,
            params={
                "format": "json",
                "lat": lat,
                "lon": lon,
                "zoom": 17,
                "addressdetails": 1,
            },
            headers={
                "User-Agent": "RainReady-App",
                "Accept-Language": "en",
            },
        )
        addr = data.get("address", {}) or {}
        area = (
            addr.get("neighbourhood")
            or addr.get("suburb")
            or addr.get("road")
            or addr.get("quarter")
            or addr.get("residential")
            or addr.get("village")
        )
        city = (
            addr.get("city")
            or addr.get("town")
            or addr.get("municipality")
            or addr.get("state_district")
            or addr.get("state")
            or fallback_city
        )
        if area and city and area.lower() != city.lower():
            place_name = f"{area}, {city}"
        elif area:
            place_name = area
        elif data.get("name"):
            place_name = f"{data['name']}, {city}" if city else str(data["name"])
        else:
            place_name = fallback_name

        subtitle = data.get("display_name", f"{lat:.4f}°N, {lon:.4f}°E")
        short_sub = ", ".join(p.strip() for p in subtitle.split(",")[:4])
        resolved = {
            "place_name": place_name,
            "city": str(city),
            "subtitle": short_sub or f"{lat:.4f}°N, {lon:.4f}°E",
        }
        GEOCODE_CACHE[cache_key] = resolved
        return resolved
    except Exception as exc:
        logger.warning("Nominatim reverse-geocode fallback for (%s, %s): %s", lat, lon, exc)
        return {
            "place_name": fallback_name,
            "city": fallback_city,
            "subtitle": f"{lat:.4f}°N, {lon:.4f}°E",
        }


async def _fetch_google_6h_forecast(
    lat: float, lng: float, weather_key: str
) -> dict[str, Any]:
    """Fetch next 6 hours from Google Maps Platform Weather API (`v1/forecast/hours:lookup`)."""
    params = {
        "key": weather_key,
        "location.latitude": lat,
        "location.longitude": lng,
        "hours": 6,
        "pageSize": 6,
    }
    payload = await _http_get_json(GOOGLE_FORECAST_HOURS_URL, params)
    forecast_hours = payload.get("forecastHours", [])[:6]
    if not forecast_hours:
        raise ValueError("Google Weather API returned empty forecastHours[].")

    hourly_records: list[dict[str, Any]] = []
    for idx, fh in enumerate(forecast_hours):
        disp_dt = fh.get("displayDateTime", {}) or {}
        raw_hour = disp_dt.get("hours")
        precip = fh.get("precipitation", {}) or {}
        qpf_mm = float((precip.get("qpf") or {}).get("quantity", 0.0) or 0.0)
        prob_pct = int((precip.get("probability") or {}).get("percent", 0) or 0)
        cond_text = ((fh.get("weatherCondition") or {}).get("description") or {}).get(
            "text", ""
        )
        tstorm_pct = int(fh.get("thunderstormProbability", 0) or 0)

        hourly_records.append(
            {
                "slot_index": idx,
                "hour_label": SLOT_LABELS[idx] if idx < len(SLOT_LABELS) else f"+{idx}h",
                "display_hour": int(raw_hour) if raw_hour is not None else None,
                "clock_label": _format_hour_clock(raw_hour),
                "mm_per_hr": round(qpf_mm, 2),
                "probability_pct": prob_pct,
                "thunderstorm_probability_pct": tstorm_pct,
                "condition_text": cond_text,
            }
        )

    first = hourly_records[0]
    return {
        "hourly_records": hourly_records,
        "condition_text": first.get("condition_text") or "Live forecast",
        "thunderstorm_probability_pct": first.get("thunderstorm_probability_pct", 0),
        "observation_time": (forecast_hours[0].get("interval") or {}).get("startTime"),
        "source": "Google Maps Platform Weather API (6h Forecast)",
    }


async def _fetch_open_meteo_6h_forecast(lat: float, lng: float) -> dict[str, Any]:
    """Automatic 6-hour fallback using Open-Meteo."""
    params = {
        "latitude": lat,
        "longitude": lng,
        "hourly": "precipitation,precipitation_probability",
        "forecast_hours": 6,
        "timezone": "auto",
    }
    payload = await _http_get_json(OPEN_METEO_URL, params)
    hourly = payload.get("hourly", {}) or {}
    times = hourly.get("time", [])[:6]
    precips = hourly.get("precipitation", [])[:6]
    probs = hourly.get("precipitation_probability", [])[:6]

    hourly_records: list[dict[str, Any]] = []
    for idx in range(min(6, len(precips))):
        iso_str = times[idx] if idx < len(times) else ""
        disp_hr: Optional[int] = None
        if "T" in iso_str:
            try:
                disp_hr = int(iso_str.split("T")[1].split(":")[0])
            except (ValueError, IndexError):
                disp_hr = None

        hourly_records.append(
            {
                "slot_index": idx,
                "hour_label": SLOT_LABELS[idx] if idx < len(SLOT_LABELS) else f"+{idx}h",
                "display_hour": disp_hr,
                "clock_label": _format_hour_clock(disp_hr),
                "mm_per_hr": round(float(precips[idx] or 0.0), 2),
                "probability_pct": int(probs[idx] or 0) if idx < len(probs) else 0,
                "thunderstorm_probability_pct": 0,
                "condition_text": "Open-Meteo Forecast",
            }
        )

    return {
        "hourly_records": hourly_records,
        "condition_text": "Open-Meteo Hourly Forecast",
        "thunderstorm_probability_pct": 0,
        "observation_time": times[0] if times else _utc_now_iso(),
        "source": "Open-Meteo Live 6h Forecast",
    }


async def _fetch_live_6h_forecast(lat: float, lng: float) -> dict[str, Any]:
    load_dotenv(override=True)
    weather_key = os.environ.get("GOOGLE_WEATHER_API_KEY", "").strip()
    if weather_key:
        try:
            return await _fetch_google_6h_forecast(lat, lng, weather_key)
        except Exception as exc:
            logger.warning("Google 6h Forecast API error (%s); falling back to Open-Meteo.", exc)
    return await _fetch_open_meteo_6h_forecast(lat, lng)


def _compute_rain_and_duration_metrics(
    zone: dict[str, Any],
    base_forecast: dict[str, Any],
    simulated_rain_mm: Optional[float],
) -> dict[str, Any]:
    meta = zone["metadata"]
    threshold_mm = float(meta["drainage_threshold_mm_per_hr"])
    sat_hours = float(meta["saturation_hours"])

    raw_hours = copy.deepcopy(base_forecast.get("hourly_records", []))
    while len(raw_hours) < 6:
        idx = len(raw_hours)
        raw_hours.append(
            {
                "slot_index": idx,
                "hour_label": SLOT_LABELS[idx],
                "display_hour": None,
                "clock_label": "",
                "mm_per_hr": 0.0,
                "probability_pct": 0,
                "thunderstorm_probability_pct": 0,
                "condition_text": "Forecast",
            }
        )

    live_baseline_now_mm = float(raw_hours[0]["mm_per_hr"])

    if simulated_rain_mm is not None:
        sim_val = round(float(simulated_rain_mm), 2)
        for i in range(min(3, len(raw_hours))):
            raw_hours[i]["mm_per_hr"] = sim_val
            raw_hours[i]["probability_pct"] = max(int(raw_hours[i].get("probability_pct", 0)), 95)
            raw_hours[i]["condition_text"] = "Simulated Heavy Cloudburst"

    for h in raw_hours:
        h["exceeds_drain_capacity"] = bool(float(h["mm_per_hr"]) >= threshold_mm)

    current_rate_mm_hr = round(float(raw_hours[0]["mm_per_hr"]), 2)
    expected_6h_total_mm = round(sum(float(h["mm_per_hr"]) for h in raw_hours), 2)
    expected_3h_total_mm = round(sum(float(h["mm_per_hr"]) for h in raw_hours[:3]), 2)

    max_continuous = 0
    curr_streak = 0
    for h in raw_hours:
        if float(h["mm_per_hr"]) > 2.0:
            curr_streak += 1
            max_continuous = max(max_continuous, curr_streak)
        else:
            curr_streak = 0
    continuous_rain_hours = float(max_continuous)

    max_exceed_streak = 0
    exceed_rates: list[float] = []
    curr_ex = 0
    curr_rates: list[float] = []
    for h in raw_hours:
        rate = float(h["mm_per_hr"])
        if rate >= threshold_mm:
            curr_ex += 1
            curr_rates.append(rate)
            if curr_ex > max_exceed_streak:
                max_exceed_streak = curr_ex
                exceed_rates = list(curr_rates)
        else:
            curr_ex = 0
            curr_rates = []

    estimated_hours_to_flood: Optional[float] = None
    saturation_volume_mm = threshold_mm * sat_hours
    if max_exceed_streak > 0 and exceed_rates:
        avg_exceed_rate = sum(exceed_rates) / len(exceed_rates)
        storm_volume_mm = avg_exceed_rate * max_exceed_streak
        if max_exceed_streak >= sat_hours or storm_volume_mm >= (saturation_volume_mm * 0.85):
            est = sat_hours * (threshold_mm / max(avg_exceed_rate, threshold_mm))
            estimated_hours_to_flood = round(max(0.5, min(sat_hours, est)), 1)

    return {
        "observed_precipitation_mm": live_baseline_now_mm,
        "observed_rain_mm": live_baseline_now_mm,
        "current_rate_mm_hr": current_rate_mm_hr,
        "effective_rain_mm": current_rate_mm_hr,
        "expected_3h_total_mm": expected_3h_total_mm,
        "expected_6h_total_mm": expected_6h_total_mm,
        "continuous_rain_hours": continuous_rain_hours,
        "sustained_over_threshold_hours": float(max_exceed_streak),
        "estimated_hours_to_flood": estimated_hours_to_flood,
        "drainage_threshold_mm_per_hr": threshold_mm,
        "saturation_hours": sat_hours,
        "hourly_forecast": raw_hours,
        "condition_text": raw_hours[0].get("condition_text") or base_forecast.get("condition_text"),
        "thunderstorm_probability_pct": base_forecast.get("thunderstorm_probability_pct", 0),
        "simulated_rain_mm": simulated_rain_mm,
        "is_simulated": simulated_rain_mm is not None,
        "unit": "mm",
        "observation_time": base_forecast.get("observation_time"),
        "source": base_forecast.get("source", "Google Maps Platform Weather API"),
        "fetched_at": _utc_now_iso(),
    }


def _build_zone_input(
    zone: dict[str, Any],
    weather: dict[str, Any],
    nearest_hotspot_context: Optional[str] = None,
) -> str:
    meta = zone["metadata"]
    reports = zone["reports"]

    if reports:
        formatted_reports = "\n".join(
            f"  - [{r['timestamp']}] {r['depth']} at {r.get('place_name', zone['name'])}: {r['description']}"
            for r in reports[-15:]
        )
    else:
        formatted_reports = "  - None reported near this location yet."

    hourly_lines = "\n".join(
        f"  - {h['hour_label']} ({h.get('clock_label') or 'forecast'}): "
        f"{h['mm_per_hr']:.2f} mm/hr | rain_prob={h['probability_pct']}% | "
        f"exceeds_drain_capacity={h['exceeds_drain_capacity']}"
        for h in weather["hourly_forecast"]
    )

    sim_banner = (
        f"SIMULATED 3-HOUR CONTINUOUS STORM ACTIVE: {weather['simulated_rain_mm']:.1f} mm/hr "
        f"injected across Now, +1h, +2h (Live baseline Now was {weather['observed_precipitation_mm']:.2f} mm/hr)."
        if weather["is_simulated"]
        else f"LIVE FORECAST TELEMETRY ({weather['source']})."
    )

    hotspot_line = (
        f"\nHistorical Hotspot Context: {nearest_hotspot_context}"
        if nearest_hotspot_context
        else f"\nHistorical Flood Note: {zone.get('historical_flood_note', meta['drainage_profile'])}"
    )

    return f"""HYPERLOCAL RAINFALL INTENSITY + DURATION TELEMETRY
Location Name: {zone['name']} ({zone['subtitle']}) [id={zone['id']}, city={zone.get('city', 'Urban')}]
Coordinates: {zone['lat']}, {zone['lng']}
Baseline Elevation: {meta['elevation_m']}m above MSL
Drainage Threshold: {meta['drainage_threshold_mm_per_hr']} mm/hr
Saturation Window: Floods if >= {meta['drainage_threshold_mm_per_hr']} mm/hr continues for {meta['saturation_hours']} hours
Impervious Surface Coverage: {meta['impervious_surface_pct']}%
Canal / Outfall Basin: {meta['canal_basin']}{hotspot_line}

6-HOUR FORECAST & PRE-COMPUTED DURATION METRICS:
- Mode: {sim_banner}
- current_rate_mm_hr: {weather['current_rate_mm_hr']:.2f} mm/hr
- expected_3h_total_mm: {weather['expected_3h_total_mm']:.2f} mm
- expected_6h_total_mm: {weather['expected_6h_total_mm']:.2f} mm
- continuous_rain_hours (> 2.0 mm/hr): {weather['continuous_rain_hours']:.1f} hours
- consecutive_hours_exceeding_threshold: {weather['sustained_over_threshold_hours']:.1f} hours
- estimated_hours_to_flood: {weather['estimated_hours_to_flood']}
- Thunderstorm Probability: {weather.get('thunderstorm_probability_pct', 0)}%

NEXT 6 HOURS BREAKDOWN:
{hourly_lines}

NEARBY CITIZEN WATERLOGGING REPORTS ({len(reports)} total):
{formatted_reports}"""


def _to_plain_dict(obj: Any) -> dict[str, float]:
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        raw = obj.model_dump()
    elif isinstance(obj, dict):
        raw = obj
    else:
        raw = dict(obj)
    return {str(k): round(float(v), 4) for k, v in raw.items()}


async def _run_stage1_decision_gate(
    v1beta_client: genai.Client,
    shared_input: str,
) -> dict[str, Any]:
    """Stage 1 — Fast Calibration Gate (`gemini-decision-1` on `beta_client.decisions.create`)."""
    decision = await asyncio.to_thread(
        v1beta_client.decisions.create,
        model=MODEL_IDS["decisions"],
        input=shared_input,
        questions={
            "escalate_alert": {
                "type": "noul",
                "instructions": "Should this location trigger a flood watch or warning based on forecast rainfall intensity, sustained duration, historical flood vulnerability, and citizen reports?",
            },
            "risk_level": {
                "type": "choice",
                "instructions": "Classify the flood risk level for this neighborhood over the current and upcoming 6-hour rainfall window.",
                "criteria": {
                    "high_danger": (
                        "Sustained heavy rain exceeding the street's drainage threshold across its saturation window, "
                        "active >=30mm/hr cloudburst, or knee-deep citizen waterlogging reports."
                    ),
                    "low_risk": (
                        "Dry or light rainfall safely below the street's drainage threshold with no hazardous waterlogging reports."
                    ),
                    "other": (
                        "Ambiguous, contradictory, or unreadable sensor/report input requiring manual review."
                    ),
                    "watch_level": (
                        "Moderate sustained rain approaching the drainage threshold, short burst near threshold, "
                        "blocked drain report, or ankle-deep curb ponding."
                    ),
                },
            },
            "severity_score": {
                "type": "score",
                "instructions": "Rate expected street waterlogging severity across 3 ordered levels from minor pooling to severe waterlogging.",
                "criteria": [
                    "No flooding or minor surface dampness; storm drains flowing well within capacity.",
                    "Localized curb/ankle-deep pooling, partial drain blockage, or runoff nearing capacity.",
                    "Severe knee-deep or impassable street waterlogging from sustained over-threshold rain.",
                ],
            },
        },
    )

    escalate_ans = decision.answers["escalate_alert"]
    risk_ans = decision.answers["risk_level"]
    severity_ans = decision.answers["severity_score"]

    escalate_prob = float(getattr(escalate_ans, "noul", 0.0))
    escalate_conf = float(getattr(escalate_ans, "confidence", 0.0))

    choice_key = str(getattr(risk_ans, "choice", "other"))
    choice_conf = float(getattr(risk_ans, "confidence", 0.0))
    choice_probs = _to_plain_dict(getattr(risk_ans, "probabilities", None))

    severity_val = float(getattr(severity_ans, "score", 0.0))
    severity_conf = float(getattr(severity_ans, "confidence", 0.0))
    severity_probs = getattr(severity_ans, "probabilities", None)
    if severity_probs is not None and not isinstance(severity_probs, list):
        severity_probs = list(severity_probs)

    choice_to_ui_risk = {
        "low_risk": "LOW",
        "watch_level": "WATCH",
        "high_danger": "HIGH",
        "other": "WATCH" if escalate_prob >= 0.50 else "LOW",
    }
    mapped_risk_level = choice_to_ui_risk.get(choice_key, "WATCH")
    calibrated_confidence_pct = int(round(max(0.0, min(1.0, choice_conf)) * 100))

    return {
        "model": MODEL_IDS["decisions"],
        "escalate_alert_prob": round(escalate_prob, 4),
        "escalate_alert_confidence": round(escalate_conf, 4),
        "risk_choice_key": choice_key,
        "mapped_risk_level": mapped_risk_level,
        "risk_choice_probabilities": choice_probs,
        "risk_choice_confidence": round(choice_conf, 4),
        "calibrated_confidence_pct": calibrated_confidence_pct,
        "severity_score": round(severity_val, 4),
        "severity_max": 2.0,
        "severity_confidence": round(severity_conf, 4),
        "severity_probabilities": [round(float(p), 4) for p in (severity_probs or [])],
        "output_tokens_billed": 0,
    }


async def _run_stage2_neon_reasoning(
    v1_client: genai.Client,
    zone: dict[str, Any],
    weather: dict[str, Any],
    shared_input: str,
    stage1_gate: dict[str, Any],
) -> tuple[Stage2ActionableReasoning, dict[str, Any], Optional[str]]:
    """Stage 2 — Actionable Duration & Forecast Reasoning (`gemini-4-neon`, `thinking_level="medium"`)."""
    meta = zone["metadata"]
    stage2_prompt = f"""{shared_input}

STAGE 1 GEMINI DECISION 1 CALIBRATED PROBABILITIES & SCORES:
- escalate_alert P(yes): {stage1_gate['escalate_alert_prob']:.4f} (confidence: {stage1_gate['escalate_alert_confidence']:.4f})
- risk_level choice: "{stage1_gate['risk_choice_key']}" -> {stage1_gate['mapped_risk_level']} (calibrated confidence: {stage1_gate['calibrated_confidence_pct']}%)
- risk_level choice probabilities: {stage1_gate['risk_choice_probabilities']}
- severity_score (0.00 minor pooling to 2.00 severe waterlogging): {stage1_gate['severity_score']:.4f} / 2.00

HEADLINE & DURATION INSTRUCTIONS FOR {zone['name']}:
- Street drainage threshold: {meta['drainage_threshold_mm_per_hr']} mm/hr; floods if sustained for {meta['saturation_hours']} hours.
- Pre-computed estimated_hours_to_flood: {weather['estimated_hours_to_flood']}.
- If risk_level is WATCH or HIGH and forecast rainfall meets/exceeds threshold, format `duration_alert_headline` strictly as:
  "Flood risk in {zone['name']} if rain continues at [X] mm/hr for [Y] hours."
- Ensure `hourly_breakdown` contains all 6 entries ("Now", "+1h", "+2h", "+3h", "+4h", "+5h") matching the 6-hour forecast."""

    response_schema = Stage2ActionableReasoning.model_json_schema()
    neon_model = MODEL_IDS["frontier_fast"]

    last_err: Optional[Exception] = None
    for attempt in range(2):
        try:
            interaction = await asyncio.to_thread(
                v1_client.interactions.create,
                model=neon_model,
                system_instruction=SYSTEM_INSTRUCTION,
                input=stage2_prompt,
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": response_schema,
                },
                generation_config={"thinking_level": "medium"},
            )
            output_text = getattr(interaction, "output_text", None)
            if not output_text:
                output_text = "".join(
                    part.text
                    for step in getattr(interaction, "steps", [])
                    if getattr(step, "type", "") == "model_output"
                    for part in getattr(step, "content", [])
                    if getattr(part, "type", "") == "text"
                )
            if not output_text:
                raise RuntimeError("Empty output_text returned from gemini-4-neon.")

            parsed = Stage2ActionableReasoning.model_validate_json(output_text)
            routing_meta = {
                "decision_model": stage1_gate["model"],
                "reasoning_model": neon_model,
                "thinking_level": "medium",
            }
            return parsed, routing_meta, getattr(interaction, "id", None)
        except Exception as exc:
            last_err = exc
            err_str = str(exc)
            if attempt == 0 and ("429" in err_str or "503" in err_str or "UNAVAILABLE" in err_str):
                await asyncio.sleep(1.5)
                continue
            break

    logger.exception("Stage 2 gemini-4-neon call failed: %s", last_err)
    raise HTTPException(
        status_code=502,
        detail=f"Live Gemini 4 Neon Interactions API call failed: {last_err}",
    )


async def _evaluate_target_record(
    target: dict[str, Any],
    simulated_rain_mm: Optional[float] = None,
    nearest_hotspot_context: Optional[str] = None,
) -> dict[str, Any]:
    v1_client, v1beta_client = _init_genai_clients()

    try:
        base_forecast = await _fetch_live_6h_forecast(target["lat"], target["lng"])
    except Exception as exc:
        logger.warning("6h forecast fetch error for %s: %s", target["id"], exc)
        if simulated_rain_mm is None:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to fetch live 6-hour weather forecast: {exc}",
            ) from exc
        base_forecast = {
            "hourly_records": [
                {
                    "slot_index": i,
                    "hour_label": SLOT_LABELS[i],
                    "display_hour": None,
                    "clock_label": "",
                    "mm_per_hr": 0.0,
                    "probability_pct": 0,
                    "thunderstorm_probability_pct": 0,
                    "condition_text": "Simulated Override",
                }
                for i in range(6)
            ],
            "condition_text": "Simulated Heavy Rain Override",
            "thunderstorm_probability_pct": 0,
            "observation_time": _utc_now_iso(),
            "source": "Weather API Unreachable (Using 3h Storm Override)",
        }

    weather_snapshot = _compute_rain_and_duration_metrics(
        zone=target,
        base_forecast=base_forecast,
        simulated_rain_mm=simulated_rain_mm,
    )
    shared_input = _build_zone_input(
        zone=target,
        weather=weather_snapshot,
        nearest_hotspot_context=nearest_hotspot_context,
    )

    try:
        stage1_gate = await _run_stage1_decision_gate(v1beta_client, shared_input)
    except Exception as exc:
        logger.exception("Stage 1 gemini-decision-1 call failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"Live Gemini Decision 1 API call failed: {exc}",
        ) from exc

    stage2_reasoning, routing_meta, interaction_id = await _run_stage2_neon_reasoning(
        v1_client=v1_client,
        zone=target,
        weather=weather_snapshot,
        shared_input=shared_input,
        stage1_gate=stage1_gate,
    )

    eval_dict = {
        "risk_level": stage1_gate["mapped_risk_level"],
        "confidence_score": stage1_gate["calibrated_confidence_pct"],
        "severity_score": stage1_gate["severity_score"],
        "escalate_alert_prob": stage1_gate["escalate_alert_prob"],
        **stage2_reasoning.model_dump(),
    }

    now_iso = _utc_now_iso()
    target["weather"] = weather_snapshot
    target["latest_evaluation"] = eval_dict
    target["decisions_telemetry"] = stage1_gate
    target["routing_info"] = routing_meta
    target["last_interaction_id"] = interaction_id
    target["review_status"] = "PENDING_REVIEW"
    target["approved"] = False
    target["evaluated_at"] = now_iso

    return {
        **eval_dict,
        "zone_id": target["id"],
        "is_custom_location": target.get("is_custom_location", False),
        "name": target["name"],
        "subtitle": target["subtitle"],
        "city": target.get("city", ""),
        "lat": target["lat"],
        "lng": target["lng"],
        "historical_flood_note": target.get("historical_flood_note", ""),
        "metadata": target["metadata"],
        "nearest_hotspot": target.get("nearest_hotspot"),
        "review_status": target["review_status"],
        "approved": target["approved"],
        "evaluated_at": now_iso,
        "weather": weather_snapshot,
        "decisions_telemetry": stage1_gate,
        "routing_info": routing_meta,
        "interaction_id": interaction_id,
        "report_count": len(target["reports"]),
    }


async def _evaluate_zone(
    zone_id: str, simulated_rain_mm: Optional[float] = None
) -> dict[str, Any]:
    if zone_id not in ZONES:
        raise HTTPException(status_code=404, detail=f"Unknown zone_id '{zone_id}'.")

    async with ZONE_LOCKS[zone_id]:
        return await _evaluate_target_record(
            target=ZONES[zone_id],
            simulated_rain_mm=simulated_rain_mm,
        )


async def _evaluate_custom_location(
    lat: float,
    lon: float,
    simulated_rain_mm: Optional[float] = None,
    is_gps: bool = False,
) -> dict[str, Any]:
    geo = await _reverse_geocode(lat, lon)
    nearest_zone, dist_km = _find_nearest_hotspot(lat, lon)
    near_hotspot = dist_km <= 1.5

    spot_key = f"loc_{round(lat, 3)}_{round(lon, 3)}".replace("-", "m").replace(".", "_")

    if near_hotspot:
        threshold_mm = float(nearest_zone["metadata"]["drainage_threshold_mm_per_hr"])
        sat_hours = float(nearest_zone["metadata"]["saturation_hours"])
        elev_m = float(nearest_zone["metadata"]["elevation_m"])
        impervious_pct = int(nearest_zone["metadata"]["impervious_surface_pct"])
        basin = nearest_zone["metadata"]["canal_basin"]
        hist_note = (
            f"{dist_km:.2f} km from {nearest_zone['name']} flood hotspot — "
            f"{nearest_zone['historical_flood_note']}"
        )
    else:
        threshold_mm = 32.0
        sat_hours = 2.0
        elev_m = float(nearest_zone["metadata"]["elevation_m"]) if dist_km < 35.0 else 15.0
        impervious_pct = 82
        basin = f"Urban storm drain network ({dist_km:.1f} km from {nearest_zone['name']})"
        hist_note = (
            f"Outside primary mapped flood hotspots (nearest: {nearest_zone['name']}, {dist_km:.1f} km away). "
            f"Standard urban storm-drain threshold (~32 mm/hr for 2.0h) applied."
        )

    nearby_reports = _reports_near_point(lat, lon, radius_km=2.0)

    async with GLOBAL_LOC_LOCK:
        existing = DYNAMIC_LOCATIONS.get(spot_key)
        spot_record: dict[str, Any] = {
            "id": spot_key,
            "city": geo["city"],
            "is_custom_location": True,
            "is_gps": is_gps,
            "name": geo["place_name"],
            "subtitle": geo["subtitle"],
            "lat": round(lat, 5),
            "lng": round(lon, 5),
            "historical_flood_note": hist_note,
            "nearest_hotspot": {
                "id": nearest_zone["id"],
                "name": nearest_zone["name"],
                "city": nearest_zone["city"],
                "distance_km": dist_km,
                "within_hotspot_radius": near_hotspot,
                "historical_flood_note": nearest_zone["historical_flood_note"],
            },
            "metadata": {
                "elevation_m": elev_m,
                "drainage_capacity_mm_hr": threshold_mm,
                "drainage_threshold_mm_per_hr": threshold_mm,
                "saturation_hours": sat_hours,
                "impervious_surface_pct": impervious_pct,
                "canal_basin": basin,
                "drainage_profile": hist_note,
            },
            "reports": nearby_reports,
            "weather": existing["weather"] if existing else None,
            "latest_evaluation": existing["latest_evaluation"] if existing else None,
            "decisions_telemetry": existing["decisions_telemetry"] if existing else None,
            "routing_info": existing["routing_info"] if existing else None,
            "last_interaction_id": existing["last_interaction_id"] if existing else None,
            "review_status": existing["review_status"] if existing else "UNASSESSED",
            "approved": existing["approved"] if existing else False,
            "evaluated_at": existing["evaluated_at"] if existing else None,
            "approved_alert": existing["approved_alert"] if existing else None,
        }
        DYNAMIC_LOCATIONS[spot_key] = spot_record

        return await _evaluate_target_record(
            target=spot_record,
            simulated_rain_mm=simulated_rain_mm,
            nearest_hotspot_context=hist_note,
        )


# ---------------------------------------------------------------------------
# FastAPI App & Routes
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RainReady — Hyperlocal GPS & Hotspot Flood Intelligence",
    description=(
        "Live GPS location & flood-hotspot warning engine powered by Google Weather API, "
        "OpenStreetMap Nominatim, Gemini Decision 1, and Gemini 4 Neon."
    ),
    version="3.1.0",
    lifespan=lifespan,
)

STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def serve_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
async def serve_favicon() -> FileResponse:
    return FileResponse(STATIC_DIR / "favicon.ico", media_type="image/x-icon")


@app.get("/api/health")
async def health_check() -> dict[str, Any]:
    load_dotenv(override=True)
    has_gemini_key = bool(os.environ.get("GEMINI_API_KEY", "").strip())
    has_weather_key = bool(os.environ.get("GOOGLE_WEATHER_API_KEY", "").strip())
    return {
        "status": "ok",
        "service": "rainready",
        "cloud_run_project": CLOUD_RUN_PROJECT,
        "models": {
            "stage1_gate": MODEL_IDS["decisions"],
            "stage2_reasoning": MODEL_IDS["frontier_fast"],
            "thinking_level": "medium",
        },
        "weather_provider": (
            "Google Maps Platform Weather API (6h Forecast)"
            if has_weather_key
            else "Open-Meteo Live 6h Forecast"
        ),
        "hotspot_count": len(ZONES),
        "active_reports": len(ALL_REPORTS),
        "gemini_key_configured": has_gemini_key,
        "google_weather_key_configured": has_weather_key,
    }


@app.get("/api/zones")
async def list_zones() -> dict[str, Any]:
    return {"zones": list(ZONES.values())}


@app.get("/api/reports")
async def list_reports() -> dict[str, Any]:
    return {"reports": ALL_REPORTS}


@app.get("/api/state")
async def get_full_state() -> dict[str, Any]:
    return {
        "zones": list(ZONES.values()),
        "dynamic_locations": list(DYNAMIC_LOCATIONS.values()),
        "reports": ALL_REPORTS,
    }


@app.post("/api/analyze/{zone_id}")
async def analyze_zone(
    zone_id: str,
    payload: Optional[AnalyzeRequest] = Body(default=None),
    simulated_rain_mm: Optional[float] = Query(default=None, ge=0.0, le=500.0),
) -> dict[str, Any]:
    override_mm = (
        payload.simulated_rain_mm
        if (payload is not None and payload.simulated_rain_mm is not None)
        else simulated_rain_mm
    )
    if zone_id in DYNAMIC_LOCATIONS:
        loc = DYNAMIC_LOCATIONS[zone_id]
        return await _evaluate_custom_location(
            lat=loc["lat"],
            lon=loc["lng"],
            simulated_rain_mm=override_mm,
            is_gps=bool(loc.get("is_gps")),
        )
    return await _evaluate_zone(zone_id=zone_id, simulated_rain_mm=override_mm)


@app.post("/api/analyze-location")
async def analyze_location(req: AnalyzeLocationRequest) -> dict[str, Any]:
    return await _evaluate_custom_location(
        lat=req.lat,
        lon=req.lon,
        simulated_rain_mm=req.simulated_rain_mm,
        is_gps=req.is_gps,
    )


@app.post("/api/report")
async def submit_report(report_req: ReportRequest) -> dict[str, Any]:
    zone_id = report_req.zone_id
    lat = report_req.lat
    lon = report_req.lon

    # Resolve coordinates from explicit {lat, lon}, known zone, or dynamic location
    if lat is None or lon is None:
        if zone_id and zone_id in ZONES:
            lat = ZONES[zone_id]["lat"]
            lon = ZONES[zone_id]["lng"]
        elif zone_id and zone_id in DYNAMIC_LOCATIONS:
            lat = DYNAMIC_LOCATIONS[zone_id]["lat"]
            lon = DYNAMIC_LOCATIONS[zone_id]["lng"]
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide either {lat, lon} or a valid zone_id to submit a report.",
            )

    # Resolve readable place_name
    if report_req.place_name and report_req.place_name.strip():
        resolved_place = report_req.place_name.strip()
    elif zone_id and zone_id in ZONES:
        resolved_place = ZONES[zone_id]["name"]
    elif zone_id and zone_id in DYNAMIC_LOCATIONS:
        resolved_place = DYNAMIC_LOCATIONS[zone_id]["name"]
    else:
        geo = await _reverse_geocode(lat, lon)
        resolved_place = geo["place_name"]

    clean_depth = report_req.depth.strip()
    clean_desc = (
        report_req.description.strip()
        or f"{clean_depth} waterlogging reported at {resolved_place}."
    )

    report_entry = {
        "id": uuid.uuid4().hex[:8],
        "zone_id": zone_id,
        "lat": round(float(lat), 5),
        "lon": round(float(lon), 5),
        "place_name": resolved_place,
        "depth": clean_depth,
        "description": clean_desc,
        "timestamp": _utc_now_iso(),
    }
    ALL_REPORTS.append(report_entry)

    # Attach to explicit hotspot or any hotspot within 1.5 km
    matched_hotspot_id: Optional[str] = None
    if zone_id and zone_id in ZONES:
        ZONES[zone_id]["reports"].append(report_entry)
        matched_hotspot_id = zone_id
    else:
        nearest_z, dist_km = _find_nearest_hotspot(lat, lon)
        if dist_km <= 1.5:
            nearest_z["reports"].append(report_entry)
            matched_hotspot_id = nearest_z["id"]

    # Immediately re-run live Gemini 4 evaluation for the active context
    if zone_id and zone_id in ZONES:
        evaluation_result = await _evaluate_zone(
            zone_id=zone_id,
            simulated_rain_mm=report_req.simulated_rain_mm,
        )
    else:
        evaluation_result = await _evaluate_custom_location(
            lat=float(lat),
            lon=float(lon),
            simulated_rain_mm=report_req.simulated_rain_mm,
            is_gps=bool(zone_id and zone_id in DYNAMIC_LOCATIONS and DYNAMIC_LOCATIONS[zone_id].get("is_gps")),
        )
        # If also adjacent to a mapped hotspot, keep its report list fresh
        if matched_hotspot_id and matched_hotspot_id != evaluation_result.get("zone_id"):
            ZONES[matched_hotspot_id]["review_status"] = "PENDING_REVIEW"

    return {
        "report": report_entry,
        "reports": ALL_REPORTS,
        "evaluation": evaluation_result,
        **evaluation_result,
    }


@app.post("/api/approve/{zone_id}")
async def approve_zone_alert(zone_id: str) -> dict[str, Any]:
    if zone_id in ZONES:
        async with ZONE_LOCKS[zone_id]:
            zone = ZONES[zone_id]
            if zone["latest_evaluation"] is None:
                raise HTTPException(
                    status_code=400,
                    detail="Cannot approve an unassessed zone.",
                )
            approved_at = _utc_now_iso()
            zone["approved"] = True
            zone["review_status"] = "APPROVED"
            zone["approved_alert"] = {
                **copy.deepcopy(zone["latest_evaluation"]),
                "zone_id": zone_id,
                "zone_name": zone["name"],
                "approved_at": approved_at,
                "evaluated_at": zone["evaluated_at"],
                "weather": copy.deepcopy(zone["weather"]),
                "decisions_telemetry": copy.deepcopy(zone["decisions_telemetry"]),
                "routing_info": copy.deepcopy(zone["routing_info"]),
                "report_count": len(zone["reports"]),
            }
            return {
                "zone_id": zone_id,
                "approved": True,
                "review_status": "APPROVED",
                "approved_alert": zone["approved_alert"],
            }

    if zone_id in DYNAMIC_LOCATIONS:
        async with GLOBAL_LOC_LOCK:
            loc = DYNAMIC_LOCATIONS[zone_id]
            if loc["latest_evaluation"] is None:
                raise HTTPException(
                    status_code=400,
                    detail="Cannot approve an unassessed location.",
                )
            approved_at = _utc_now_iso()
            loc["approved"] = True
            loc["review_status"] = "APPROVED"
            loc["approved_alert"] = {
                **copy.deepcopy(loc["latest_evaluation"]),
                "zone_id": zone_id,
                "zone_name": loc["name"],
                "approved_at": approved_at,
                "evaluated_at": loc["evaluated_at"],
                "weather": copy.deepcopy(loc["weather"]),
                "decisions_telemetry": copy.deepcopy(loc["decisions_telemetry"]),
                "routing_info": copy.deepcopy(loc["routing_info"]),
                "report_count": len(loc["reports"]),
            }
            return {
                "zone_id": zone_id,
                "approved": True,
                "review_status": "APPROVED",
                "approved_alert": loc["approved_alert"],
            }

    raise HTTPException(status_code=404, detail=f"Unknown zone_id '{zone_id}'.")


@app.post("/api/reset")
async def reset_demo_state() -> dict[str, Any]:
    global ZONES
    fresh = _build_initial_zones()
    for zid, data in fresh.items():
        ZONES[zid] = data
    DYNAMIC_LOCATIONS.clear()
    ALL_REPORTS.clear()
    return {
        "status": "reset",
        "zones": list(ZONES.values()),
        "dynamic_locations": [],
        "reports": [],
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
