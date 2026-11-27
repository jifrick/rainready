# RainReady — Kochi Hyperlocal Flood Alert Engine

Built following the **Build with Gemini 4: Hackathon Builder Guide** and packaged for **Google Cloud Run** (`geminisprints26blr-3834`) on port `8080`.

## Multi-Model Gemini 4 + Decisions API Loop
RainReady implements the two-stage hackathon architecture recommended in the Builder Guide:
1. **Stage 1 — Calibrated Gate & Rubric Scoring (`gemini-decision-1` via `v1beta` Decisions API):**
   Evaluates three atomic questions in a single forward pass with **zero output tokens**:
   - `has_waterlogging_hazard` (`noul` yes/no probability + confidence)
   - `risk_route` (`choice` across `low_clear`, `watch_advisory`, `high_alert`, and explicit `other` catch-all)
   - `drainage_stress` (`score` ordered 3-level rubric)
2. **Stage 2 — Confidence-Gated Gemini 4 Synthesis (`client.aio.interactions.create`):**
   - Escalates to **`gemini-4-neon`** (`thinking_level="high"`, with automatic free-tier fallback to **`gemini-4-hydrogen`** `thinking_level="high"`) whenever Decision 1 detects elevated waterlogging probability (`P > 0.50`) or uncertainty (`confidence < 0.75`).
   - Fast-routes routine dry/clear weather to **`gemini-4-hydrogen`** (`thinking_level="low"`).
   - Uses native Pydantic JSON Schema `response_format` and stateful `previous_interaction_id` chaining per neighborhood zone, with zero custom `temperature` / `top_p` / `top_k` / `thinking_budget` fields.
3. **Human-in-the-Loop Gate:**
   - Every AI assessment enters `PENDING_REVIEW` first; responders review calibrated probabilities, confidence, and signals before clicking **Approve & Broadcast** (`POST /api/approve/{zone_id}`) to publish to the public Resident View.

## Installed Agent Skills (`.agents/skills/`)
- `google-gemini/gemini-skills`: `gemini-api-dev`, `gemini-live-api-dev`
- `google/skills`: `cloud-run-basics`, `gcloud`, `gemini-api`, `finding-google-skills`

## Local Setup (< 2 Minutes)
1. Paste your Google AI Studio and Google Weather API keys into `.env` (excluded via `.gitignore` and `.dockerignore`):
   ```ini
   GEMINI_API_KEY=your_ai_studio_key
   GOOGLE_WEATHER_API_KEY=your_google_weather_api_key
   ```
   *(Uses **Google Maps Platform Weather API** `v1/currentConditions:lookup` for current QPF, 24-hour antecedent rainfall, and thunderstorm probability when `GOOGLE_WEATHER_API_KEY` is set, with automatic fallback to Open-Meteo.)*
2. Install dependencies (`google-genai>=3.2.0`) and start on port `8080`:
   ```bash
   pip install -U -r requirements.txt
   python -m uvicorn main:app --host 0.0.0.0 --port 8080
   ```
3. Open `http://localhost:8080`.
