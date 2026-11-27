import asyncio
import copy
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Literal, Optional

import httpx
from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from models import MODEL_IDS, create_clients

# Load environment variables from local .env file
load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rainready")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

CLOUD_RUN_PROJECT = "geminisprints26blr-3834"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

SYSTEM_INSTRUCTION = """You are RainReady's Hyperlocal Flood Risk Engine for Kochi, Kerala (Emergency Operations & Civic Resilience System).
Synthesize live Open-Meteo precipitation telemetry, calibrated Gemini Decision 1 probability/rubric scores, neighborhood elevation/drainage terrain characteristics, and timestamped crowd-sourced resident reports into an actionable flood assessment.

Calibration Guidelines:
- LOW: Minimal or light rainfall with no significant waterlogging reports; storm drains operating within capacity.
- WATCH: Moderate rainfall, blocked drain reports, ankle-deep ponding in low-lying pockets, or elevated runoff approaching canal/culvert capacity.
- HIGH: Heavy rainfall / cloudburst conditions (e.g., >=30mm/hr), knee-deep waterlogging, multiple corroborating reports in sub-grade underpasses or low-elevation marshland corridors, or severe drainage obstruction during rain.

Output Requirements:
- Cite exact observable causal drivers in `visible_signals` (mm/hr rainfall, Decision 1 hazard probability/rubric score, elevation/canal bottleneck, and resident report details).
- Keep `responder_summary` concise, operational, and tailored to municipal disaster-management human reviewers deciding whether to approve public broadcast.
- Provide 2 to 3 concrete, plain-language `resident_actions` tailored to commuters and households in this specific Kochi neighborhood.
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
            "decisions_telemetry": None,
            "routing_info": None,
            "last_interaction_id": None,
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
            "decisions_telemetry": None,
            "routing_info": None,
            "last_interaction_id": None,
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
            "decisions_telemetry": None,
            "routing_info": None,
            "last_interaction_id": None,
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
            "decisions_telemetry": None,
            "routing_info": None,
            "last_interaction_id": None,
            "review_status": "UNASSESSED",
            "approved": False,
            "evaluated_at": None,
            "approved_alert": None,
        },
    }


ZONES: dict[str, dict[str, Any]] = _build_initial_zones()
ZONE_LOCKS: dict[str, asyncio.Lock] = {zid: asyncio.Lock() for zid in ZONES}

# Shared connection-pooled HTTP client for Cloud Run concurrency efficiency
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


def _get_clients():
    """Return (stable_v1_client, v1beta_client) using os.environ['GEMINI_API_KEY']."""
    try:
        return create_clients()
    except KeyError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "GEMINI_API_KEY is missing. Add GEMINI_API_KEY=<your_key> to the local .env "
                "file in the workspace (or Cloud Run environment variables) to run live Gemini evaluations."
            ),
        ) from exc


async def _fetch_open_meteo(lat: float, lng: float) -> dict[str, Any]:
    """Fetch live current precipitation and rain from Open-Meteo API."""
    params = {
        "latitude": lat,
        "longitude": lng,
        "current": "precipitation,rain",
    }
    if _http_client is not None:
        resp = await _http_client.get(OPEN_METEO_URL, params=params)
        resp.raise_for_status()
        payload = resp.json()
    else:
        async with httpx.AsyncClient(timeout=10.0) as fallback_client:
            resp = await fallback_client.get(OPEN_METEO_URL, params=params)
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


def _build_shared_context_block(
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
        else (
            f"LIVE TELEMETRY MODE: Active precipitation is {weather['observed_precipitation_mm']:.1f} mm "
            f"(rain: {weather['observed_rain_mm']:.1f} mm) observed at {weather.get('observation_time', 'current')}."
        )
    )

    return f"""KOCHI NEIGHBORHOOD FLOOD TELEMETRY SNAPSHOT
Zone ID: {zone['id']}
Zone Name: {zone['name']} ({zone['subtitle']})
Coordinates: {zone['lat']}, {zone['lng']}
Mean Elevation: {meta['elevation_m']}m above MSL
Design Storm Drainage Capacity: ~{meta['drainage_capacity_mm_hr']} mm/hr
Impervious Surface Coverage: {meta['impervious_surface_pct']}%
Receiving Canal / Basin: {meta['canal_basin']}
Terrain & Hydrological Profile: {meta['drainage_profile']}

WEATHER TELEMETRY:
{sim_note}
Effective Rainfall Intensity: {weather['effective_rain_mm']:.1f} mm/hr

CROWD-SOURCED RESIDENT WATERLOGGING REPORTS ({len(reports)} total):
{formatted_reports}"""


async def _run_decision_gate(
    beta_client: Any,
    shared_context: str,
) -> dict[str, Any]:
    """Stage 1: Fast calibrated gating & rubric scoring via Gemini Decision 1 (0 output tokens).

    Follows the Gemini 4 Hackathon Builder Guide (Pages 6-8):
    - Evaluates yes/no (`noul`), `choice` (with explicit `other` catch-all key), and ordered `score` rubric in 1 pass.
    """
    decision = await beta_client.aio.decisions.create(
        model=MODEL_IDS["decisions"],
        input=shared_context,
        questions={
            "has_waterlogging_hazard": {
                "type": "noul",
                "instructions": (
                    "Is active or imminent street-level waterlogging or drainage surcharge "
                    "threatening safe neighborhood transit right now?"
                ),
            },
            "risk_route": {
                "type": "choice",
                "instructions": "Which hyperlocal flood alert tier fits the current weather, terrain, and ground reports?",
                "criteria": {
                    "high_alert": (
                        "Severe rainfall/cloudburst (>=30mm/hr), knee-deep waterlogging reports, "
                        "or overwhelmed sub-grade culvert/canal backflow."
                    ),
                    "low_clear": (
                        "Dry or light rainfall well within storm-drain capacity with no hazardous ponding reports."
                    ),
                    "other": (
                        "Ambiguous, contradictory, or unreadable sensor/report data requiring human inspection."
                    ),
                    "watch_advisory": (
                        "Moderate rainfall, blocked drain report, ankle-deep ponding, or rising runoff approaching capacity."
                    ),
                },
            },
            "drainage_stress": {
                "type": "score",
                "instructions": "Rate the hydrological stress on this neighborhood's storm drainage network.",
                "criteria": [
                    "Drains clear and operating comfortably within design capacity.",
                    "Localized ponding, drain obstruction, or moderate runoff approaching culvert limits.",
                    "Severe surcharge, knee-deep roadway flooding, or cloudburst runoff exceeding capacity.",
                ],
            },
        },
    )

    hazard_ans = decision.answers["has_waterlogging_hazard"]
    route_ans = decision.answers["risk_route"]
    stress_ans = decision.answers["drainage_stress"]

    choice_map = {
        "low_clear": "LOW",
        "watch_advisory": "WATCH",
        "high_alert": "HIGH",
        "other": "WATCH",
    }
    winning_choice = getattr(route_ans, "choice", "other")
    recommended_risk = choice_map.get(winning_choice, "WATCH")

    raw_probs = getattr(route_ans, "probabilities", {}) or {}
    if hasattr(raw_probs, "model_dump"):
        raw_probs = raw_probs.model_dump()
    elif not isinstance(raw_probs, dict):
        raw_probs = dict(raw_probs)

    hazard_prob = float(getattr(hazard_ans, "noul", 0.0))
    route_conf = float(getattr(route_ans, "confidence", 0.0))
    hazard_conf = float(getattr(hazard_ans, "confidence", 0.0))
    stress_score = float(getattr(stress_ans, "score", 0.0))
    stress_conf = float(getattr(stress_ans, "confidence", 0.0))

    # Escalate to deeper reasoning when hazard is present or Decision 1 certainty < 0.75 (Builder Guide p.8)
    escalate = bool(
        route_conf < 0.75
        or hazard_prob > 0.50
        or winning_choice in ("high_alert", "watch_advisory", "other")
    )

    calibrated_conf_pct = int(
        round(max(0.0, min(1.0, (route_conf * 0.6 + hazard_conf * 0.25 + stress_conf * 0.15))) * 100)
    )

    return {
        "model": MODEL_IDS["decisions"],
        "hazard_probability": round(hazard_prob, 4),
        "hazard_confidence": round(hazard_conf, 4),
        "winning_choice": winning_choice,
        "recommended_risk_level": recommended_risk,
        "choice_probabilities": {k: round(float(v), 4) for k, v in raw_probs.items()},
        "route_confidence": round(route_conf, 4),
        "drainage_stress_score": round(stress_score, 4),
        "calibrated_confidence_pct": calibrated_conf_pct,
        "escalate_reasoning": escalate,
        "output_tokens_billed": 0,
    }


async def _run_gemini4_synthesis(
    client: Any,
    shared_context: str,
    decision_gate: Optional[dict[str, Any]],
    previous_interaction_id: Optional[str],
) -> tuple[GeminiRiskAssessment, dict[str, Any], Optional[str]]:
    """Stage 2: Structured synthesis using Gemini 4 Interactions API (`client.aio.interactions.create`).

    Follows Gemini 4 Hackathon Builder Guide (Pages 5-6 & Page 11):
    - Escalates to `gemini-4-neon` (`thinking_level="high"`) on elevated risk / low certainty, with
      automatic fallback to `gemini-4-hydrogen` (`thinking_level="high"`) on free-tier AI Studio keys.
    - Uses `gemini-4-hydrogen` (`thinking_level="low"`) for fast routine clear-weather evaluations.
    - Uses thinking_level ("low" or "high") with native JSON Schema response_format only.
    """
    escalate = bool(decision_gate["escalate_reasoning"]) if decision_gate else True
    if escalate:
        candidate_chain = [
            (MODEL_IDS["frontier_fast"], "high", "Escalated via Decision 1 gate (hazard / uncertainty)"),
            (MODEL_IDS["fast_workhorse"], "high", "Gemini 4 Hydrogen high-thinking synthesis"),
        ]
    else:
        candidate_chain = [
            (MODEL_IDS["fast_workhorse"], "low", "Fast-routed via Decision 1 high-confidence clear gate"),
        ]

    gate_context = ""
    if decision_gate is not None:
        gate_context = f"""

CALIBRATED GEMINI DECISION 1 GATE SIGNALS (0-output-token forward pass):
- Waterlogging Hazard Probability P(yes): {decision_gate['hazard_probability']:.2f} (confidence: {decision_gate['hazard_confidence']:.2f})
- Routed Tier: {decision_gate['winning_choice']} -> {decision_gate['recommended_risk_level']} (route confidence: {decision_gate['route_confidence']:.2f})
- Tier Probabilities: {decision_gate['choice_probabilities']}
- Drainage Stress Rubric Score: {decision_gate['drainage_stress_score']:.2f} / 2.00
- Calibrated Baseline Confidence: {decision_gate['calibrated_confidence_pct']}%

Ensure your structured assessment aligns with the physical evidence and calibrated probabilities above."""

    full_input = f"{shared_context}{gate_context}\n\nProduce the structured flood risk JSON assessment."
    schema_dict = GeminiRiskAssessment.model_json_schema()

    last_err: Optional[Exception] = None
    for model_id, thinking_level, route_reason in candidate_chain:
        for attempt in range(2):
            try:
                kwargs: dict[str, Any] = {
                    "model": model_id,
                    "system_instruction": SYSTEM_INSTRUCTION,
                    "input": full_input,
                    "response_format": {
                        "type": "text",
                        "mime_type": "application/json",
                        "schema": schema_dict,
                    },
                    "generation_config": {
                        "thinking_level": thinking_level,
                    },
                }
                if previous_interaction_id:
                    kwargs["previous_interaction_id"] = previous_interaction_id

                interaction = await client.aio.interactions.create(**kwargs)
                output_text = getattr(interaction, "output_text", None)
                if not output_text:
                    # Fallback extraction across model_output steps per Builder Guide p.6
                    output_text = "".join(
                        part.text
                        for step in getattr(interaction, "steps", [])
                        if getattr(step, "type", "") == "model_output"
                        for part in getattr(step, "content", [])
                        if getattr(part, "type", "") == "text"
                    )
                if not output_text:
                    raise RuntimeError(f"Empty output_text from {model_id}")

                assessment = GeminiRiskAssessment.model_validate_json(output_text)
                routing_meta = {
                    "synthesis_model": model_id,
                    "thinking_level": thinking_level,
                    "route_reason": route_reason,
                    "decision_model": decision_gate["model"] if decision_gate else None,
                    "chained_previous_interaction": bool(previous_interaction_id),
                }
                return assessment, routing_meta, getattr(interaction, "id", None)
            except Exception as exc:
                last_err = exc
                err_str = str(exc)
                # If previous_interaction_id expired (free-tier 1-day retention), retry statelessly once
                if previous_interaction_id and ("interaction" in err_str.lower() or "404" in err_str):
                    previous_interaction_id = None
                    continue
                if attempt == 0 and ("429" in err_str or "503" in err_str or "UNAVAILABLE" in err_str):
                    await asyncio.sleep(1.2)
                    continue
                logger.info("Model %s (%s) unavailable or failed (%s); trying next if available.", model_id, thinking_level, err_str)
                break

    logger.exception("Gemini 4 synthesis failed: %s", last_err)
    raise HTTPException(
        status_code=502,
        detail=f"Live Gemini 4 Interactions API call failed: {last_err}",
    )


async def _evaluate_zone(
    zone_id: str, simulated_rain_mm: Optional[float] = None
) -> dict[str, Any]:
    if zone_id not in ZONES:
        raise HTTPException(status_code=404, detail=f"Unknown zone_id '{zone_id}'.")

    client, beta_client = _get_clients()

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

        shared_context = _build_shared_context_block(
            zone=zone,
            weather=weather_snapshot,
            simulated_rain_mm=simulated_rain_mm,
        )

        # 2. Stage 1: Gemini Decision 1 calibrated probability & rubric gate (v1beta)
        decision_telemetry: Optional[dict[str, Any]] = None
        try:
            decision_telemetry = await _run_decision_gate(beta_client, shared_context)
        except Exception as dec_exc:
            logger.warning("Gemini Decision 1 gate warning (continuing with Gemini 4 escalation): %s", dec_exc)

        # 3. Stage 2: Gemini 4 Interactions API Structured Output synthesis (v1)
        assessment, routing_meta, interaction_id = await _run_gemini4_synthesis(
            client=client,
            shared_context=shared_context,
            decision_gate=decision_telemetry,
            previous_interaction_id=zone.get("last_interaction_id"),
        )
        eval_dict = assessment.model_dump()

        # 4. Enter PENDING_REVIEW Human-in-the-Loop Gate
        now_iso = _utc_now_iso()
        zone["weather"] = weather_snapshot
        zone["latest_evaluation"] = eval_dict
        zone["decisions_telemetry"] = decision_telemetry
        zone["routing_info"] = routing_meta
        zone["last_interaction_id"] = interaction_id
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
            "decisions_telemetry": decision_telemetry,
            "routing_info": routing_meta,
            "interaction_id": interaction_id,
            "report_count": len(zone["reports"]),
        }


# ---------------------------------------------------------------------------
# FastAPI Application & Endpoints
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RainReady — Kochi Hyperlocal Flood Alert Engine",
    description=(
        "Live Open-Meteo + Gemini Decision 1 + Gemini 4 (Hydrogen/Neon) neighborhood "
        "flood risk engine with human-in-the-loop broadcast gate."
    ),
    version="2.0.0",
    lifespan=lifespan,
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
        "models": {
            "decision_gate": MODEL_IDS["decisions"],
            "workhorse": MODEL_IDS["fast_workhorse"],
            "frontier_escalation": MODEL_IDS["frontier_fast"],
        },
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
    clean_desc = (
        report_req.description.strip()
        or f"{clean_depth} reported by resident in {ZONES[zone_id]['name']}."
    )

    report_entry = {
        "id": uuid.uuid4().hex[:8],
        "zone_id": zone_id,
        "depth": clean_depth,
        "description": clean_desc,
        "timestamp": _utc_now_iso(),
    }
    ZONES[zone_id]["reports"].append(report_entry)

    # Immediately re-run live Decision 1 + Gemini 4 evaluation with the new resident report included
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


@app.post("/api/reset")
async def reset_demo_state() -> dict[str, Any]:
    global ZONES
    fresh = _build_initial_zones()
    for zid, data in fresh.items():
        ZONES[zid] = data
    return {"status": "reset", "zones": list(ZONES.values())}


if __name__ == "__main__":
    import uvicorn

    # Cloud Run Container Contract: bind 0.0.0.0 on injected $PORT (default 8080)
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
