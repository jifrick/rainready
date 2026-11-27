import asyncio
import copy
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

import httpx
from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Load environment variables from local .env file
load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rainready")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

GEMINI_MODEL = "gemini-2.5-flash"
CLOUD_RUN_PROJECT = "geminisprints26blr-3834"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

SYSTEM_INSTRUCTION = """You are RainReady's Hyperlocal Flood Risk Engine for Kochi, Kerala (Emergency Operations & Civic Resilience System).
Your job is to synthesize live meteorological precipitation data, neighborhood elevation/drainage terrain characteristics, and timestamped crowd-sourced resident reports to produce an accurate, calibrated flood risk evaluation.

Calibration Guidelines:
- LOW: Minimal or light rainfall with no significant waterlogging reports; storm drains operating within capacity.
- WATCH: Moderate rainfall, blocked drain reports, ankle-deep ponding in low-lying pockets, or elevated runoff approaching canal/culvert capacity.
- HIGH: Heavy rainfall / cloudburst conditions (e.g., >=30mm/hr), knee-deep waterlogging, multiple corroborating reports in sub-grade underpasses or low-elevation marshland corridors, or severe drainage obstruction during rain.

Confidence Score Calibration (0-100):
- Base confidence on signal agreement, sensor freshness, terrain vulnerability clarity, and number/consistency of resident reports.
- Explain specific causal factors clearly in `visible_signals` (citing exact mm rainfall, elevation/canal context, and resident report details).
- Keep `responder_summary` concise, operational, and tailored to municipal disaster-management reviewers deciding whether to broadcast an alert.
- Provide 2 to 3 concrete, plain-language `resident_actions` tailored to the specific Kochi zone (e.g., alternate routes, avoiding sub-grade underpasses, two-wheeler precautions, clearing house-front grates, or routine vigilance when LOW).
"""


# ---------------------------------------------------------------------------
# Pydantic Models (Structured Outputs & API Contracts)
# ---------------------------------------------------------------------------
class GeminiRiskAssessment(BaseModel):
    risk_level: Literal["LOW", "WATCH", "HIGH"] = Field(
        ...,
        description="Overall flood risk classification: LOW, WATCH, or HIGH.",
    )
    confidence_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Calibrated confidence score from 0 to 100.",
    )
    visible_signals: list[str] = Field(
        ...,
        min_length=2,
        max_length=6,
        description="Specific observable drivers behind this assessment (rainfall mm, terrain/canal bottleneck, resident reports).",
    )
    responder_summary: str = Field(
        ...,
        description="Operational summary for human reviewers explaining why this risk level and confidence score were assigned.",
    )
    resident_actions: list[str] = Field(
        ...,
        min_length=2,
        max_length=3,
        description="2 to 3 clear, practical next steps in plain language for neighborhood residents and commuters.",
    )


class AnalyzeRequest(BaseModel):
    simulated_rain_mm: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=500.0,
        description="Optional simulated rainfall intensity override (mm/hr) for responder cloudburst testing.",
    )


class ReportRequest(BaseModel):
    zone_id: str = Field(..., description="Target Kochi neighborhood zone ID.")
    depth: str = Field(
        ...,
        min_length=2,
        max_length=80,
        description="Reported waterlogging depth or condition (e.g., Ankle-Deep, Knee-Deep, Blocked Drain).",
    )
    description: str = Field(
        default="",
        max_length=500,
        description="Additional context from the resident.",
    )
    simulated_rain_mm: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=500.0,
        description="Preserves active cloudburst simulation override if toggled in UI.",
    )


# ---------------------------------------------------------------------------
# Preloaded In-Memory Kochi Neighborhood Zones (Zero Static AI Responses)
# ---------------------------------------------------------------------------
def _build_initial_zones() -> dict[str, dict[str, Any]]:
    return {
        "south_railway_underpass": {
            "id": "south_railway_underpass",
            "name": "South Railway Underpass",
            "subtitle": "Ernakulam Junction South · Karshaka Rd Sub-Grade Corridor",
            "lat": 9.9696,
            "lng": 76.2908,
            "metadata": {
                "elevation_m": 1.8,
                "drainage_capacity_mm_hr": 18,
                "impervious_surface_pct": 89,
                "canal_basin": "Perandoor Canal (180m east)",
                "drainage_profile": (
                    "Depressed railway box-culvert underpass collecting rapid sheet runoff from "
                    "Karshaka Road and Ernakulam South yard; gravity outfall slows sharply during "
                    "high tide or Perandoor Canal surcharge."
                ),
            },
            "polygon": {
                "type": "Feature",
                "properties": {"zone_id": "south_railway_underpass"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [76.2878, 9.9672],
                            [76.2938, 9.9672],
                            [76.2942, 9.9718],
                            [76.2882, 9.9721],
                            [76.2878, 9.9672],
                        ]
                    ],
                },
            },
            "reports": [],
            "weather": None,
            "latest_evaluation": None,
            "review_status": "UNASSESSED",
            "approved": False,
            "evaluated_at": None,
            "approved_alert": None,
        },
        "panampilly_nagar": {
            "id": "panampilly_nagar",
            "name": "Panampilly Nagar",
            "subtitle": "Main Avenue & Cross-Grid Residential Marsh Basin",
            "lat": 9.9605,
            "lng": 76.2975,
            "metadata": {
                "elevation_m": 2.4,
                "drainage_capacity_mm_hr": 24,
                "impervious_surface_pct": 78,
                "canal_basin": "Thevara–Perandoor Canal (adjacent)",
                "drainage_profile": (
                    "Reclaimed low-lying estuarine residential grid with covered slab drains along "
                    "Main Avenue; susceptible to backwater ponding in internal cross-roads when "
                    "Thevara-Perandoor Canal water levels rise."
                ),
            },
            "polygon": {
                "type": "Feature",
                "properties": {"zone_id": "panampilly_nagar"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [76.2936, 9.9554],
                            [76.3022, 9.9554],
                            [76.3025, 9.9652],
                            [76.2940, 9.9652],
                            [76.2936, 9.9554],
                        ]
                    ],
                },
            },
            "reports": [],
            "weather": None,
            "latest_evaluation": None,
            "review_status": "UNASSESSED",
            "approved": False,
            "evaluated_at": None,
            "approved_alert": None,
        },
        "mg_road_padma": {
            "id": "mg_road_padma",
            "name": "MG Road / Padma Junction",
            "subtitle": "Central Commercial Spine · Mullassery Canal Catchment",
            "lat": 9.9806,
            "lng": 76.2836,
            "metadata": {
                "elevation_m": 2.1,
                "drainage_capacity_mm_hr": 20,
                "impervious_surface_pct": 94,
                "canal_basin": "Mullassery Canal (250m south-west)",
                "drainage_profile": (
                    "High-density paved commercial corridor along Kochi Metro viaduct; roadside "
                    "box drains experience siltation and debris blockage, causing rapid curb-to-shopfront "
                    "flooding near Padma Junction."
                ),
            },
            "polygon": {
                "type": "Feature",
                "properties": {"zone_id": "mg_road_padma"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [76.2798, 9.9762],
                            [76.2874, 9.9762],
                            [76.2878, 9.9848],
                            [76.2801, 9.9850],
                            [76.2798, 9.9762],
                        ]
                    ],
                },
            },
            "reports": [],
            "weather": None,
            "latest_evaluation": None,
            "review_status": "UNASSESSED",
            "approved": False,
            "evaluated_at": None,
            "approved_alert": None,
        },
        "edappally_toll": {
            "id": "edappally_toll",
            "name": "Edappally Toll Junction",
            "subtitle": "NH-66 / NH-544 Arterial Convergence · Edappally Thodu",
            "lat": 10.0261,
            "lng": 76.3083,
            "metadata": {
                "elevation_m": 3.6,
                "drainage_capacity_mm_hr": 28,
                "impervious_surface_pct": 91,
                "canal_basin": "Edappally Thodu (320m west)",
                "drainage_profile": (
                    "Major national highway and flyover ramp catchment channeling heavy asphalt "
                    "runoff into service-road drains near Lulu Mall; utility duct bottlenecks slow "
                    "discharge toward Edappally Thodu during intense bursts."
                ),
            },
            "polygon": {
                "type": "Feature",
                "properties": {"zone_id": "edappally_toll"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [76.3038, 10.0218],
                            [76.3128, 10.0218],
                            [76.3131, 10.0304],
                            [76.3041, 10.0306],
                            [76.3038, 10.0218],
                        ]
                    ],
                },
            },
            "reports": [],
            "weather": None,
            "latest_evaluation": None,
            "review_status": "UNASSESSED",
            "approved": False,
            "evaluated_at": None,
            "approved_alert": None,
        },
    }


ZONES: dict[str, dict[str, Any]] = _build_initial_zones()
ZONE_LOCKS: dict[str, asyncio.Lock] = {zid: asyncio.Lock() for zid in ZONES}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_gemini_client() -> genai.Client:
    """Instantiate Google GenAI client strictly using os.environ['GEMINI_API_KEY']."""
    load_dotenv(override=True)
    try:
        api_key = os.environ["GEMINI_API_KEY"].strip()
        if not api_key:
            raise KeyError("GEMINI_API_KEY is empty")
    except KeyError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "GEMINI_API_KEY is missing. Add GEMINI_API_KEY=<your_key> to the local .env "
                "file in the workspace (or Cloud Run environment variables) to run live Gemini evaluations."
            ),
        ) from exc
    return genai.Client(api_key=api_key)


async def _fetch_open_meteo(lat: float, lng: float) -> dict[str, Any]:
    """Fetch live current precipitation and rain from Open-Meteo API."""
    params = {
        "latitude": lat,
        "longitude": lng,
        "current": "precipitation,rain",
    }
    async with httpx.AsyncClient(timeout=10.0) as http_client:
        resp = await http_client.get(OPEN_METEO_URL, params=params)
        resp.raise_for_status()
        payload = resp.json()

    current = payload.get("current", {})
    units = payload.get("current_units", {})
    return {
        "observed_precipitation_mm": float(current.get("precipitation", 0.0) or 0.0),
        "observed_rain_mm": float(current.get("rain", 0.0) or 0.0),
        "unit": units.get("precipitation", "mm"),
        "observation_time": current.get("time"),
        "source": "Open-Meteo Live API",
    }


def _build_gemini_prompt(
    zone: dict[str, Any],
    weather: dict[str, Any],
    simulated_rain_mm: Optional[float],
) -> str:
    meta = zone["metadata"]
    reports = zone["reports"]

    if reports:
        formatted_reports = "\n".join(
            f"  - [{r['timestamp']}] Severity/Type: {r['depth']} | Details: {r['description'] or 'Reported via 1-tap resident alert'}"
            for r in reports[-15:]
        )
    else:
        formatted_reports = "  - None reported in this zone yet."

    sim_note = (
        f"ACTIVE CLOUDBURST OVERRIDE: {simulated_rain_mm:.1f} mm/hr simulated rainfall intensity "
        f"(Live Open-Meteo baseline was {weather['observed_precipitation_mm']:.1f} mm). "
        f"Treat {simulated_rain_mm:.1f} mm/hr as the active real-time rainfall intensity hitting this zone right now."
        if simulated_rain_mm is not None
        else f"LIVE TELEMETRY MODE: No simulation override. Active precipitation is {weather['observed_precipitation_mm']:.1f} mm "
             f"(rain: {weather['observed_rain_mm']:.1f} mm) observed at {weather.get('observation_time', 'current')}."
    )

    return f"""Evaluate hyperlocal flood risk right now for the following Kochi neighborhood zone:

ZONE PROFILE:
- Zone ID: {zone['id']}
- Name: {zone['name']} ({zone['subtitle']})
- Coordinates: {zone['lat']}, {zone['lng']}
- Mean Elevation: {meta['elevation_m']} meters above MSL
- Design Storm Drainage Capacity: ~{meta['drainage_capacity_mm_hr']} mm/hr
- Impervious Surface Coverage: {meta['impervious_surface_pct']}%
- Receiving Canal / Basin: {meta['canal_basin']}
- Hydrological & Terrain Notes: {meta['drainage_profile']}

WEATHER & PRECIPITATION TELEMETRY:
- {sim_note}
- Effective Rainfall Intensity to Evaluate: {weather['effective_rain_mm']:.1f} mm/hr

CROWD-SOURCED RESIDENT WATERLOGGING REPORTS ({len(reports)} total):
{formatted_reports}

Return a strictly structured JSON assessment matching the schema."""


def _call_gemini_sync(prompt: str) -> GeminiRiskAssessment:
    client = _get_gemini_client()
    last_err: Optional[Exception] = None

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=GeminiRiskAssessment,
                    temperature=0.2,
                ),
            )
            if getattr(response, "parsed", None) and isinstance(
                response.parsed, GeminiRiskAssessment
            ):
                return response.parsed
            if not response.text:
                raise RuntimeError("Gemini API returned an empty response payload.")
            return GeminiRiskAssessment.model_validate_json(response.text)
        except HTTPException:
            raise
        except Exception as exc:
            last_err = exc
            err_str = str(exc)
            # Retry transient overload/rate-limit errors briefly
            if attempt < 2 and ("429" in err_str or "503" in err_str or "UNAVAILABLE" in err_str):
                import time

                time.sleep(1.5 * (attempt + 1))
                continue
            break

    logger.exception("Gemini evaluation failed: %s", last_err)
    raise HTTPException(
        status_code=502,
        detail=f"Live Gemini API call failed: {last_err}",
    )


async def _evaluate_zone(
    zone_id: str, simulated_rain_mm: Optional[float] = None
) -> dict[str, Any]:
    if zone_id not in ZONES:
        raise HTTPException(status_code=404, detail=f"Unknown zone_id '{zone_id}'.")

    async with ZONE_LOCKS[zone_id]:
        zone = ZONES[zone_id]

        # 1. Fetch live precipitation from Open-Meteo
        try:
            live_weather = await _fetch_open_meteo(zone["lat"], zone["lng"])
        except httpx.HTTPError as exc:
            logger.warning("Open-Meteo fetch error for %s: %s", zone_id, exc)
            if simulated_rain_mm is None:
                raise HTTPException(
                    status_code=502,
                    detail=f"Failed to fetch live weather from Open-Meteo: {exc}",
                ) from exc
            live_weather = {
                "observed_precipitation_mm": 0.0,
                "observed_rain_mm": 0.0,
                "unit": "mm",
                "observation_time": _utc_now_iso(),
                "source": "Open-Meteo Unreachable (Using Cloudburst Override)",
            }

        effective_rain = (
            float(simulated_rain_mm)
            if simulated_rain_mm is not None
            else max(
                live_weather["observed_precipitation_mm"],
                live_weather["observed_rain_mm"],
            )
        )
        weather_snapshot = {
            **live_weather,
            "simulated_rain_mm": simulated_rain_mm,
            "is_simulated": simulated_rain_mm is not None,
            "effective_rain_mm": effective_rain,
            "fetched_at": _utc_now_iso(),
        }

        # 2. Query gemini-2.5-flash via Structured Outputs
        prompt = _build_gemini_prompt(zone, weather_snapshot, simulated_rain_mm)
        assessment = await asyncio.to_thread(_call_gemini_sync, prompt)
        eval_dict = assessment.model_dump()

        # 3. Store in PENDING_REVIEW human-in-the-loop gate
        now_iso = _utc_now_iso()
        zone["weather"] = weather_snapshot
        zone["latest_evaluation"] = eval_dict
        zone["review_status"] = "PENDING_REVIEW"
        zone["approved"] = False
        zone["evaluated_at"] = now_iso

        return {
            **eval_dict,
            "zone_id": zone_id,
            "review_status": zone["review_status"],
            "approved": zone["approved"],
            "evaluated_at": now_iso,
            "weather": weather_snapshot,
            "report_count": len(zone["reports"]),
        }


# ---------------------------------------------------------------------------
# FastAPI Application & Endpoints
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RainReady — Kochi Hyperlocal Flood Alert Engine",
    description="Live Open-Meteo + Gemini 2.5 Flash neighborhood flood risk engine with human-in-the-loop broadcast gate.",
    version="1.0.0",
)

STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def serve_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health_check() -> dict[str, Any]:
    load_dotenv(override=True)
    has_key = bool(os.environ.get("GEMINI_API_KEY", "").strip())
    return {
        "status": "ok",
        "service": "rainready",
        "cloud_run_project": CLOUD_RUN_PROJECT,
        "model": GEMINI_MODEL,
        "gemini_key_configured": has_key,
    }


@app.get("/api/zones")
async def list_zones() -> dict[str, Any]:
    return {"zones": list(ZONES.values())}


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
    return await _evaluate_zone(zone_id=zone_id, simulated_rain_mm=override_mm)


@app.post("/api/report")
async def submit_report(report_req: ReportRequest) -> dict[str, Any]:
    zone_id = report_req.zone_id
    if zone_id not in ZONES:
        raise HTTPException(status_code=404, detail=f"Unknown zone_id '{zone_id}'.")

    clean_depth = report_req.depth.strip()
    clean_desc = report_req.description.strip() or f"{clean_depth} reported by resident in {ZONES[zone_id]['name']}."

    report_entry = {
        "id": uuid.uuid4().hex[:8],
        "zone_id": zone_id,
        "depth": clean_depth,
        "description": clean_desc,
        "timestamp": _utc_now_iso(),
    }
    ZONES[zone_id]["reports"].append(report_entry)

    # Immediately re-run live Gemini evaluation with the new resident report included
    evaluation_result = await _evaluate_zone(
        zone_id=zone_id,
        simulated_rain_mm=report_req.simulated_rain_mm,
    )
    return {
        "report": report_entry,
        "evaluation": evaluation_result,
        **evaluation_result,
    }


@app.post("/api/approve/{zone_id}")
async def approve_zone_alert(zone_id: str) -> dict[str, Any]:
    if zone_id not in ZONES:
        raise HTTPException(status_code=404, detail=f"Unknown zone_id '{zone_id}'.")

    async with ZONE_LOCKS[zone_id]:
        zone = ZONES[zone_id]
        if zone["latest_evaluation"] is None:
            raise HTTPException(
                status_code=400,
                detail="Cannot approve an unassessed zone. Run /api/analyze/{zone_id} first.",
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
            "report_count": len(zone["reports"]),
        }

        return {
            "zone_id": zone_id,
            "approved": True,
            "review_status": "APPROVED",
            "approved_alert": zone["approved_alert"],
        }


@app.post("/api/reset")
async def reset_demo_state() -> dict[str, Any]:
    global ZONES
    fresh = _build_initial_zones()
    for zid, data in fresh.items():
        ZONES[zid] = data
    return {"status": "reset", "zones": list(ZONES.values())}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
