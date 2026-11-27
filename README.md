# RainReady — Hyperlocal Flood Alert Engine (Kochi)

Hyperlocal neighborhood flood-alert web application packaged for **Google Cloud Run** (`geminisprints26blr-3834`), built with **FastAPI**, **Open-Meteo**, **Leaflet.js**, and live **Google Gen AI SDK** (`google-genai` · `gemini-2.5-flash`) Structured Outputs.

## Architecture & Core Rules
- **100% Live Gemini Evaluations:** Every risk level (`LOW`, `WATCH`, `HIGH`), calibrated confidence score (`0–100`), observable signal breakdown (`visible_signals`), responder summary (`responder_summary`), and resident guidance (`resident_actions`) is generated dynamically via `client.aio.models.generate_content` using Pydantic `response_schema`.
- **Zero Key Exposure:** `GEMINI_API_KEY` is read strictly on the backend via `os.environ["GEMINI_API_KEY"]` (loaded locally from `.env`, which is excluded in `.gitignore` and `.dockerignore`).
- **Human-in-the-Loop Broadcast Gate:** AI evaluations enter `PENDING_REVIEW` first; clicking **Approve & Broadcast** (`POST /api/approve/{zone_id}`) publishes verified alerts to the Resident View.
- **4 Preloaded Kochi Zones:** South Railway Underpass, Panampilly Nagar, MG Road / Padma Junction, and Edappally Toll Junction.

## Google Agent Skills (`google/skills`)
Installed in `.agents/skills/` from [google/skills](https://github.com/google/skills):
- `gemini-api` — Unified `google-genai` async (`client.aio`) & Pydantic Structured Output patterns
- `cloud-run-basics` — Cloud Run container runtime contract (`0.0.0.0`, `$PORT`, PID 1 `SIGTERM`, non-root)
- `gcloud` — Leaf-level syntax validation, explicit `--project` / `--region` scoping, and non-interactive guardrails
- `finding-google-skills` — On-demand Google skill catalog discovery

## Local Development

1. Add your API key to `.env`:
   ```ini
   GEMINI_API_KEY=your_gemini_api_key
   ```
2. Install dependencies and run on port `8080`:
   ```bash
   pip install -r requirements.txt
   python -m uvicorn main:app --host 0.0.0.0 --port 8080
   ```
3. Open `http://localhost:8080`.

## Deploying to Google Cloud Run (`geminisprints26blr-3834`)

Following the `cloud-run-basics` and `gcloud` agent skill guidelines:
1. Validate leaf command flags first: `gcloud help run deploy`
2. Deploy from source using the included `Dockerfile` (passing `GEMINI_API_KEY` securely via Secret Manager or runtime environment variables—never baked into the image):
   ```bash
   gcloud run deploy rainready \
     --source . \
     --project=geminisprints26blr-3834 \
     --region=asia-south1 \
     --allow-unauthenticated \
     --set-env-vars="GEMINI_API_KEY=YOUR_KEY" \
     --quiet
   ```
