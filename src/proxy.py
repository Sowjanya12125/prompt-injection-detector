"""
proxy.py - Prompt Injection Detection Proxy (Hybrid: ML + LLM Judge)
Sits between the client and any LLM API, routing every user message
through the hybrid pipeline before it reaches the model:

    User Prompt
         |
    ML Classifier (embedding-based, <5ms)   <- src/classifier_v2.py
         |
    Risk Score
         |
      < LOW_THRESHOLD   -> ALLOW directly, forward to target LLM
      LOW..HIGH         -> LLM Judge (secondary review)   <- src/llm_judge.py
      > HIGH_THRESHOLD  -> BLOCK directly, reject request
         |
    ALLOW -> forward   REVIEW -> hold for human review   BLOCK -> reject

Usage:
    uvicorn src.proxy:app --reload --port 8000

Then send requests to http://localhost:8000/v1/chat/completions
instead of directly to OpenAI/Anthropic — same request format,
injections get blocked before they reach the LLM.

GPU note: the LLM judge (Mistral-7B) requires a CUDA GPU. If none is
available, set ENABLE_LLM_JUDGE=false — the proxy will still run on the
ML layer alone, and any prompt that would have been escalated is
returned as REVIEW instead of silently allowed or blocked (fail-safe,
not fail-open).
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from src.classifier_v2 import load_model as load_ml_model, predict as ml_predict

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────
TARGET_URL = os.getenv("TARGET_LLM_URL", "https://api.openai.com/v1/chat/completions")
API_KEY = os.getenv("OPENAI_API_KEY", "")
LOW_THRESHOLD = float(os.getenv("LOW_THRESHOLD", "0.45"))
HIGH_THRESHOLD = float(os.getenv("HIGH_THRESHOLD", "0.75"))
ENABLE_LLM_JUDGE = os.getenv("ENABLE_LLM_JUDGE", "true").lower() == "true"

LOG_PATH = Path("logs")
LOG_PATH.mkdir(exist_ok=True)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH / "proxy.log"),
    ],
)
logger = logging.getLogger(__name__)

# ── Load ML layer (always required — this is the fast gate on every request) ──
embed_model, clf = load_ml_model()
logger.info("ML classifier (embedding) loaded.")
logger.info(f"Routing thresholds: low={LOW_THRESHOLD}, high={HIGH_THRESHOLD}")

# ── Check LLM judge availability (optional — GPU dependent) ────────────────────
llm_judge_available = False
if ENABLE_LLM_JUDGE:
    try:
        import torch
        if torch.cuda.is_available():
            llm_judge_available = True
            logger.info("LLM judge enabled — GPU detected, Mistral-7B will load lazily on first borderline request.")
        else:
            logger.warning("ENABLE_LLM_JUDGE=true but no CUDA GPU detected. "
                            "Borderline requests will be marked REVIEW instead of getting a second opinion.")
    except ImportError:
        logger.warning("ENABLE_LLM_JUDGE=true but torch/transformers not installed. "
                        "Borderline requests will be marked REVIEW instead of getting a second opinion.")
else:
    logger.info("LLM judge disabled (ENABLE_LLM_JUDGE=false). Running ML-only — borderline requests get REVIEW.")

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Prompt Injection Detection Proxy",
    description="Real-time hybrid (ML + LLM judge) adversarial prompt detection sitting between clients and LLM APIs.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response models ─────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "gpt-3.5-turbo"
    messages: list[Message]
    temperature: float = 0.7
    max_tokens: int = 500


# ── Hybrid routing ───────────────────────────────────────────────────────────
def route_prompt(text: str) -> dict:
    """
    Runs the hybrid pipeline on a single prompt. Always returns a dict with:
    text, verdict (ALLOW/REVIEW/BLOCK), source (ML/LLM), is_injection,
    ml_risk_score, threshold, action, label, confidence, and reasoning
    (only present when source == "LLM").

    is_injection/label/action/confidence/threshold are kept for backward
    compatibility with callers written against the single-classifier proxy.
    """
    ml_result = ml_predict(text, embed_model, clf)
    risk_score = ml_result["confidence"]

    if risk_score < LOW_THRESHOLD:
        verdict, source, reasoning = "ALLOW", "ML", None
    elif risk_score > HIGH_THRESHOLD:
        verdict, source, reasoning = "BLOCK", "ML", None
    elif llm_judge_available:
        try:
            from src.llm_judge import judge as llm_judge_fn
            judge_result = llm_judge_fn(text)
            verdict = judge_result["verdict"]
            source = "LLM"
            reasoning = judge_result.get("reasoning")
        except Exception as e:
            logger.error(f"LLM judge call failed, falling back to REVIEW: {e}")
            verdict, source, reasoning = "REVIEW", "ML", f"LLM judge error: {e}"
    else:
        # No GPU / judge disabled — fail-safe, not fail-open: an ambiguous
        # prompt with no second opinion available is held for human review,
        # never silently allowed.
        verdict, source, reasoning = "REVIEW", "ML", "LLM judge unavailable (no GPU) — held for human review."

    result = {
        "text": text,
        "verdict": verdict,
        "source": source,
        "ml_risk_score": risk_score,
        "reasoning": reasoning,
        # Backward-compatible fields
        "confidence": risk_score,
        "threshold": {"low": LOW_THRESHOLD, "high": HIGH_THRESHOLD},
        "is_injection": verdict == "BLOCK",
        "label": {"BLOCK": "injection", "ALLOW": "benign", "REVIEW": "review"}[verdict],
        "action": verdict.lower(),
    }
    return result


def scan_messages(messages: list[Message]) -> dict | None:
    """
    Scan all user messages in a conversation. Returns the first non-ALLOW
    result (BLOCK or REVIEW) found, else None.
    """
    for msg in messages:
        if msg.role == "user":
            result = route_prompt(msg.content)
            if result["verdict"] != "ALLOW":
                return result
    return None


# ── Request logger ───────────────────────────────────────────────────────────
def log_request(request_id: str, detection: dict | None, verdict: str, latency_ms: float):
    entry = {
        "request_id": request_id,
        "timestamp": datetime.utcnow().isoformat(),
        "verdict": verdict,
        "latency_ms": round(latency_ms, 2),
        "detection": detection,
    }
    log_file = LOG_PATH / "detections.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

    if verdict == "REVIEW":
        review_file = LOG_PATH / "review_queue.jsonl"
        with open(review_file, "a") as f:
            f.write(json.dumps(entry) + "\n")


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "Prompt Injection Detection Proxy",
        "version": "2.0.0",
        "status": "running",
        "architecture": "hybrid (embedding classifier + LLM judge)",
        "threshold": {"low": LOW_THRESHOLD, "high": HIGH_THRESHOLD},
        "llm_judge_available": llm_judge_available,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "all-MiniLM-L6-v2 + Mistral-7B-Instruct-v0.2 (hybrid)",
        "threshold": {"low": LOW_THRESHOLD, "high": HIGH_THRESHOLD},
        "llm_judge_available": llm_judge_available,
    }


@app.post("/v1/chat/completions")
async def proxy_chat(request: ChatRequest):
    """
    Drop-in replacement for OpenAI's /v1/chat/completions.
    Routes all user messages through the hybrid pipeline:
      - ALLOW  -> forwarded to the target LLM
      - REVIEW -> held for human review (202, not forwarded — fail-safe)
      - BLOCK  -> rejected (400, not forwarded)
    """
    request_id = str(uuid.uuid4())[:8]
    start = time.time()

    logger.info(f"[{request_id}] Received request — {len(request.messages)} messages")

    detection = scan_messages(request.messages)
    latency = (time.time() - start) * 1000

    if detection and detection["verdict"] == "BLOCK":
        logger.warning(
            f"[{request_id}] BLOCKED — source={detection['source']} "
            f"risk={detection['ml_risk_score']:.4f} | text='{detection['text'][:60]}...'"
        )
        log_request(request_id, detection, verdict="BLOCK", latency_ms=latency)
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "type": "prompt_injection_detected",
                    "code": "injection_blocked",
                    "message": "This request was blocked by the prompt injection detector.",
                    "request_id": request_id,
                    "confidence": detection["ml_risk_score"],
                    "source": detection["source"],
                    "verdict": detection["verdict"],
                    "reasoning": detection["reasoning"],
                    "threshold": detection["threshold"],
                }
            },
        )

    if detection and detection["verdict"] == "REVIEW":
        logger.warning(
            f"[{request_id}] HELD FOR REVIEW — source={detection['source']} "
            f"risk={detection['ml_risk_score']:.4f} | text='{detection['text'][:60]}...'"
        )
        log_request(request_id, detection, verdict="REVIEW", latency_ms=latency)
        return JSONResponse(
            status_code=202,
            content={
                "status": {
                    "type": "prompt_pending_review",
                    "code": "review_required",
                    "message": "This request was ambiguous and has been held for human review, not forwarded to the LLM.",
                    "request_id": request_id,
                    "confidence": detection["ml_risk_score"],
                    "source": detection["source"],
                    "reasoning": detection["reasoning"],
                }
            },
        )

    # ── ALLOW — forward to LLM ───────────────────────────────────────────────
    logger.info(f"[{request_id}] ALLOWED — forwarding to LLM")
    log_request(request_id, detection=None, verdict="ALLOW", latency_ms=latency)

    if not API_KEY:
        return {
            "id": f"mock-{request_id}",
            "object": "chat.completion",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "[MOCK] No API key set. Request passed injection check successfully.",
                },
                "finish_reason": "stop",
                "index": 0,
            }],
            "proxy_meta": {
                "verdict": "ALLOW",
                "scan_latency_ms": round(latency, 2),
            },
        }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                TARGET_URL,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json=request.model_dump(),
            )
            result = response.json()
            result["proxy_meta"] = {
                "verdict": "ALLOW",
                "scan_latency_ms": round(latency, 2),
            }
            return result
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="LLM API request timed out.")
        except Exception as e:
            logger.error(f"[{request_id}] LLM API error: {e}")
            raise HTTPException(status_code=502, detail=f"LLM API error: {str(e)}")


@app.post("/detect")
async def detect_only(request: Request):
    """
    Standalone detection endpoint — classify a prompt without forwarding.
    Useful for testing the detector directly.

    Body: { "text": "your prompt here" }
    """
    body = await request.json()
    text = body.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="'text' field is required.")

    result = route_prompt(text)
    return result


@app.get("/stats")
def stats():
    """Return detection statistics from the log file, broken down by verdict and source."""
    log_file = LOG_PATH / "detections.jsonl"
    if not log_file.exists():
        return {"total_requests": 0, "blocked": 0, "allowed": 0, "review": 0, "block_rate": 0}

    entries = []
    with open(log_file) as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    total = len(entries)
    blocked = sum(1 for e in entries if e.get("verdict") == "BLOCK")
    review = sum(1 for e in entries if e.get("verdict") == "REVIEW")
    allowed = total - blocked - review
    llm_calls = sum(1 for e in entries if (e.get("detection") or {}).get("source") == "LLM")

    return {
        "total_requests": total,
        "blocked": blocked,
        "allowed": allowed,
        "review": review,
        "block_rate": round(blocked / total, 4) if total > 0 else 0,
        "llm_call_rate": round(llm_calls / total, 4) if total > 0 else 0,
        "avg_latency_ms": round(
            sum(e.get("latency_ms", 0) for e in entries) / total, 2
        ) if total > 0 else 0,
    }


@app.get("/demo", response_class=FileResponse)
def demo_ui():
    """Serve the live demo UI."""
    demo_path = Path("demo/index.html")
    if not demo_path.exists():
        raise HTTPException(status_code=404, detail="Demo UI not found.")
    return FileResponse(demo_path)
