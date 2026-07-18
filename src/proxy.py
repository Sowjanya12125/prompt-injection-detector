"""
proxy.py - Prompt Injection Detection Proxy
Sits between the client and any LLM API, intercepting and
classifying every user message before it reaches the model.

Usage:
    uvicorn src.proxy:app --reload --port 8000

Then send requests to http://localhost:8000/v1/chat/completions
instead of directly to OpenAI/Anthropic — same request format, 
injections get blocked before they reach the LLM.
"""

import json
import logging
import os
import pickle
import time
import uuid
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
TARGET_URL      = os.getenv("TARGET_LLM_URL", "https://api.openai.com/v1/chat/completions")
API_KEY         = os.getenv("OPENAI_API_KEY", "")
BLOCK_THRESHOLD = float(os.getenv("BLOCK_THRESHOLD", "0.65"))
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

MODEL_DIR       = Path("models")
CLF_PATH        = MODEL_DIR / "embedding_classifier.pkl"
LOG_PATH        = Path("logs")
LOG_PATH.mkdir(exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH / "proxy.log"),
    ],
)
logger = logging.getLogger(__name__)

# ── Load model ────────────────────────────────────────────────────────────────
def load_classifier():
    if not CLF_PATH.exists():
        raise FileNotFoundError(
            f"Classifier not found at {CLF_PATH}. "
            "Run: python src/classifier_v2.py"
        )
    with open(CLF_PATH, "rb") as f:
        clf = pickle.load(f)
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info(f"Classifier loaded from {CLF_PATH}")
    logger.info(f"Embedding model: {EMBEDDING_MODEL}")
    logger.info(f"Block threshold: {BLOCK_THRESHOLD}")
    return embed_model, clf

embed_model, clf = load_classifier()

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Prompt Injection Detection Proxy",
    description="Real-time adversarial prompt detection sitting between clients and LLM APIs.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request/Response models ───────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "gpt-3.5-turbo"
    messages: list[Message]
    temperature: float = 0.7
    max_tokens: int = 500

# ── Detection logic ───────────────────────────────────────────────────────────
def detect_injection(text: str) -> dict:
    """Run the classifier on a single text. Returns detection result."""
    cleaned = text.strip().lower()
    embedding = embed_model.encode(
        [cleaned],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    prob = float(clf.predict_proba(embedding)[0][1])
    is_injection = prob >= BLOCK_THRESHOLD

    return {
        "text": text,
        "confidence": round(prob, 4),
        "is_injection": is_injection,
        "threshold": BLOCK_THRESHOLD,
    }

def scan_messages(messages: list[Message]) -> dict | None:
    """
    Scan all user messages in a conversation.
    Returns detection result for the first injection found, else None.
    """
    for msg in messages:
        if msg.role == "user":
            result = detect_injection(msg.content)
            if result["is_injection"]:
                return result
    return None

# ── Request logger ────────────────────────────────────────────────────────────
def log_request(request_id: str, detection: dict | None, blocked: bool, latency_ms: float):
    entry = {
        "request_id": request_id,
        "timestamp": datetime.utcnow().isoformat(),
        "blocked": blocked,
        "latency_ms": round(latency_ms, 2),
        "detection": detection,
    }
    log_file = LOG_PATH / "detections.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "Prompt Injection Detection Proxy",
        "version": "1.0.0",
        "status": "running",
        "threshold": BLOCK_THRESHOLD,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": EMBEDDING_MODEL, "threshold": BLOCK_THRESHOLD}


@app.post("/v1/chat/completions")
async def proxy_chat(request: ChatRequest):
    """
    Drop-in replacement for OpenAI's /v1/chat/completions.
    Scans all user messages → blocks injections → forwards safe requests to LLM.
    """
    request_id = str(uuid.uuid4())[:8]
    start = time.time()

    logger.info(f"[{request_id}] Received request — {len(request.messages)} messages")

    # ── Scan ──────────────────────────────────────────────────────────────────
    detection = scan_messages(request.messages)
    latency = (time.time() - start) * 1000

    if detection:
        logger.warning(
            f"[{request_id}] BLOCKED — confidence={detection['confidence']:.4f} | "
            f"text='{detection['text'][:60]}...'"
        )
        log_request(request_id, detection, blocked=True, latency_ms=latency)

        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "type": "prompt_injection_detected",
                    "code": "injection_blocked",
                    "message": "This request was blocked by the prompt injection detector.",
                    "request_id": request_id,
                    "confidence": detection["confidence"],
                    "threshold": BLOCK_THRESHOLD,
                }
            },
        )

    # ── Forward to LLM ────────────────────────────────────────────────────────
    logger.info(f"[{request_id}] ALLOWED — forwarding to LLM")
    log_request(request_id, detection=None, blocked=False, latency_ms=latency)

    if not API_KEY:
        # No API key set — return a mock response for testing
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
                "injection_detected": False,
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
                "injection_detected": False,
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

    result = detect_injection(text)
    return {
        "text": text,
        "label": "injection" if result["is_injection"] else "benign",
        "confidence": result["confidence"],
        "is_injection": result["is_injection"],
        "threshold": BLOCK_THRESHOLD,
        "action": "block" if result["is_injection"] else "allow",
    }


@app.get("/stats")
def stats():
    """Return detection statistics from the log file."""
    log_file = LOG_PATH / "detections.jsonl"
    if not log_file.exists():
        return {"total_requests": 0, "blocked": 0, "allowed": 0, "block_rate": 0}

    entries = []
    with open(log_file) as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    total   = len(entries)
    blocked = sum(1 for e in entries if e.get("blocked"))
    allowed = total - blocked

    return {
        "total_requests": total,
        "blocked": blocked,
        "allowed": allowed,
        "block_rate": round(blocked / total, 4) if total > 0 else 0,
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
