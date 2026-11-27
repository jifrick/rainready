import asyncio
import copy
import logging
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

# Load GEMINI_API_KEY from local .env at startup
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
    load_dotenv()
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
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

SYSTEM_INSTRUCTION = """You are RainReady's Hyperlocal Flood Actionable Reasoning Engine for Kochi, Kerala.
Given live Open-Meteo rainfall telemetry, neighborhood elevation and drainage metadata, timestamped resident reports, and calibrated probabilities/scores from Gemini Decision 1:
1. List exact observable signal drivers in `visible_signals` (citing exact rainfall rate in mm/hr, zone elevation/drainage metadata, resident reports, and calibrated Decision 1 signals).
2. Write a concise operational verification note in `responder_summary` for municipal emergency management human reviewers.
3. Provide 2 to 3 clear, practical next steps in `resident_actions` tailored to residents and commuters in this Kochi zone (e.g., moving parked vehicles, avoiding sub-grade underpasses, clearing storm grates, or normal transit readiness).
"""


# ---------------------------------------------------------------------------
# Native Structured Output Schema & Request Models
# ---------------------------------------------------------------------------
class Stage2ActionableReasoning(BaseModel):
    visible_signals: list[str] = Field(
        ...,
        min_length=2,
        max_length=6,
        description="Exact observable flood risk signal drivers (rainfall rate, elevation/drainage, resident reports).",
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


class AnalyzeRequest(BaseModel):
    simulated_rain_mm: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=500.0,
        description="Optional simulated rainfall intensity override (mm/hr) from the UI.",
    )


class ReportRequest(BaseModel):
    zone_id: str = Field(..., description="Target Kochi neighborhood zone ID.")
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
        description="Optional active cloudburst override if enabled in UI.",
    )


# ---------------------------------------------------------------------------
# 4 Kochi Neighborhood Zones (In-Memory State — Zero Static AI Responses)
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


async def _fetch_open_meteo(lat: float, lng: float) -> dict[str, Any]:
    """Fetch live precipitation (current=precipitation,rain) from Open-Meteo."""
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


def _build_zone_input(
    zone: dict[str, Any],
    weather: dict[str, Any],
    simulated_rain_mm: Optional[float],
) -> str:
    meta = zone["metadata"]
    reports = zone["reports"]

    if reports:
        formatted_reports = "\n".join(
            f"  - [{r['timestamp']}] {r['depth']}: {r['description']}"
            for r in reports[-15:]
        )
    else:
        formatted_reports = "  - None reported in this zone yet."

    rain_mode = (
        f"ACTIVE CLOUDBURST OVERRIDE: {simulated_rain_mm:.1f} mm/hr simulated rainfall "
        f"(Live Open-Meteo baseline: {weather['observed_precipitation_mm']:.1f} mm). "
        f"Evaluate {simulated_rain_mm:.1f} mm/hr as the current live rainfall intensity."
        if simulated_rain_mm is not None
        else (
            f"LIVE OPEN-METEO TELEMETRY: precipitation={weather['observed_precipitation_mm']:.1f} mm, "
            f"rain={weather['observed_rain_mm']:.1f} mm (observed at {weather.get('observation_time', 'now')})."
        )
    )

    return f"""KOCHI HYPERLOCAL FLOOD TELEMETRY
Zone: {zone['name']} ({zone['subtitle']}) [id={zone['id']}]
Coordinates: {zone['lat']}, {zone['lng']}
Baseline Elevation: {meta['elevation_m']}m above MSL
Storm Drainage Design Capacity: ~{meta['drainage_capacity_mm_hr']} mm/hr
Impervious Surface Coverage: {meta['impervious_surface_pct']}%
Canal / Outfall Basin: {meta['canal_basin']}
Terrain & Drainage Metadata: {meta['drainage_profile']}

WEATHER SIGNAL:
{rain_mode}
Effective Rainfall Rate: {weather['effective_rain_mm']:.1f} mm/hr

RESIDENT WATERLOGGING REPORTS ({len(reports)} total):
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
    """Stage 1 — Fast Calibration Gate (`gemini-decision-1` on `beta_client.decisions.create`).

    Evaluates `escalate_alert` (noul), `risk_level` (choice with `other` catch-all),
    and `severity_score` (3-level ordered score rubric) in one forward pass (0 output tokens).
    """
    decision = await asyncio.to_thread(
        v1beta_client.decisions.create,
        model=MODEL_IDS["decisions"],
        input=shared_input,
        questions={
            "escalate_alert": {
                "type": "noul",
                "instructions": "Should this zone trigger a flood watch or warning?",
            },
            "risk_level": {
                "type": "choice",
                "instructions": "Classify the flood risk level for this Kochi neighborhood zone right now.",
                "criteria": {
                    "high_danger": (
                        "Heavy rainfall or cloudburst (>=30mm/hr), knee-deep waterlogging reports, "
                        "or severe sub-grade underpass/canal surcharge."
                    ),
                    "low_risk": (
                        "Dry or light rainfall well within storm drainage capacity and no hazardous waterlogging reports."
                    ),
                    "other": (
                        "Ambiguous, contradictory, or unreadable sensor/report input requiring manual review."
                    ),
                    "watch_level": (
                        "Moderate rainfall, blocked drain report, ankle-deep waterlogging, or rising canal backwater."
                    ),
                },
            },
            "severity_score": {
                "type": "score",
                "instructions": "Rate waterlogging severity across 3 ordered levels from minor pooling to severe waterlogging.",
                "criteria": [
                    "No flooding or minor surface dampness; storm drains flowing freely.",
                    "Localized curb/ankle-deep pooling or partial drain blockage requiring caution.",
                    "Severe knee-deep or impassable street waterlogging exceeding drainage capacity.",
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

    # Extract calibrated confidence (0-100%) directly from decision.answers
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
    shared_input: str,
    stage1_gate: dict[str, Any],
) -> tuple[Stage2ActionableReasoning, dict[str, Any], Optional[str]]:
    """Stage 2 — Actionable Reasoning (`gemini-4-neon` on `client.interactions.create` with `thinking_level="medium"`).

    Uses native JSON Schema Structured Outputs (`response_format`) to generate `visible_signals`,
    `responder_summary`, and `resident_actions`.
    """
    stage2_prompt = f"""{shared_input}

STAGE 1 GEMINI DECISION 1 CALIBRATED PROBABILITIES & SCORES:
- escalate_alert P(yes): {stage1_gate['escalate_alert_prob']:.4f} (confidence: {stage1_gate['escalate_alert_confidence']:.4f})
- risk_level choice: "{stage1_gate['risk_choice_key']}" -> {stage1_gate['mapped_risk_level']} (calibrated confidence: {stage1_gate['calibrated_confidence_pct']}%)
- risk_level choice probabilities: {stage1_gate['risk_choice_probabilities']}
- severity_score (0.00 minor pooling to 2.00 severe waterlogging): {stage1_gate['severity_score']:.4f} / 2.00

Generate `visible_signals`, `responder_summary`, and `resident_actions` consistent with these calibrated signals."""

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


async def _evaluate_zone(
    zone_id: str, simulated_rain_mm: Optional[float] = None
) -> dict[str, Any]:
    if zone_id not in ZONES:
        raise HTTPException(status_code=404, detail=f"Unknown zone_id '{zone_id}'.")

    v1_client, v1beta_client = _init_genai_clients()

    async with ZONE_LOCKS[zone_id]:
        zone = ZONES[zone_id]

        # 1. Fetch live Open-Meteo precipitation
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

        shared_input = _build_zone_input(zone, weather_snapshot, simulated_rain_mm)

        # 2. Stage 1 — Fast Calibration Gate (gemini-decision-1 on beta_client.decisions.create)
        try:
            stage1_gate = await _run_stage1_decision_gate(v1beta_client, shared_input)
        except Exception as exc:
            logger.exception("Stage 1 gemini-decision-1 call failed: %s", exc)
            raise HTTPException(
                status_code=502,
                detail=f"Live Gemini Decision 1 API call failed: {exc}",
            ) from exc

        # 3. Stage 2 — Actionable Reasoning (gemini-4-neon on client.interactions.create, thinking_level="medium")
        stage2_reasoning, routing_meta, interaction_id = await _run_stage2_neon_reasoning(
            v1_client=v1_client,
            shared_input=shared_input,
            stage1_gate=stage1_gate,
        )

        # Combine Stage 1 calibrated risk/confidence/severity with Stage 2 structured reasoning
        eval_dict = {
            "risk_level": stage1_gate["mapped_risk_level"],
            "confidence_score": stage1_gate["calibrated_confidence_pct"],
            "severity_score": stage1_gate["severity_score"],
            "escalate_alert_prob": stage1_gate["escalate_alert_prob"],
            **stage2_reasoning.model_dump(),
        }

        # 4. Hold in PENDING_REVIEW until responder clicks Approve & Broadcast
        now_iso = _utc_now_iso()
        zone["weather"] = weather_snapshot
        zone["latest_evaluation"] = eval_dict
        zone["decisions_telemetry"] = stage1_gate
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
            "decisions_telemetry": stage1_gate,
            "routing_info": routing_meta,
            "interaction_id": interaction_id,
            "report_count": len(zone["reports"]),
        }


# ---------------------------------------------------------------------------
# FastAPI App & Routes
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RainReady — Kochi Hyperlocal Flood Alert Engine",
    description=(
        "Two-stage live flood alert workflow pairing Gemini Decision 1 (calibrated gate) "
        "with Gemini 4 Neon (thinking_level='medium' structured reasoning) for Cloud Run."
    ),
    version="2.1.0",
    lifespan=lifespan,
)

STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def serve_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health_check() -> dict[str, Any]:
    load_dotenv()
    has_key = bool(os.environ.get("GEMINI_API_KEY", "").strip())
    return {
        "status": "ok",
        "service": "rainready",
        "cloud_run_project": CLOUD_RUN_PROJECT,
        "models": {
            "stage1_gate": MODEL_IDS["decisions"],
            "stage2_reasoning": MODEL_IDS["frontier_fast"],
            "thinking_level": "medium",
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

    # Immediately re-run both gemini-decision-1 and gemini-4-neon so confidence and risk update live
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

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
