"""
Token Counter — Web API (Project #1, web version)

Wraps the same token-counting logic from main.py as a FastAPI endpoint,
so a frontend can call it over HTTP instead of the command line.

Run with:
    uvicorn api:app --reload

Then open index.html in your browser (just double-click it, no server needed
for the frontend itself since it's a static file).
"""

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

try:
    import tiktoken
except ImportError:
    sys.exit("tiktoken is not installed. Run: py -m pip install tiktoken")


MODEL_PRICES = {
    "gpt-4o": 2.50,
    "gpt-4o-mini": 0.15,
    "claude-sonnet-4-6": 3.00,
    "claude-haiku-4-5": 1.00,
    "gemini-2.0-flash": 0.10,
}

app = FastAPI(title="Token Counter API")

# Allows index.html (opened as a local file, origin "null") to call this API.
# Fine for a learning project on localhost — would be locked down for
# anything you actually deploy publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_encoder = None


def get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


class CountRequest(BaseModel):
    text: str
    model: str = "gpt-4o-mini"
    peek: int = 0


class CountResponse(BaseModel):
    chars: int
    words: int
    tokens: int
    ratio: float
    cost: float
    model: str
    peek_tokens: list[str]


@app.get("/")
def serve_frontend():
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "message": "Token Counter API is running"}


@app.get("/models")
def list_models():
    return {"models": list(MODEL_PRICES.keys())}


@app.post("/count", response_model=CountResponse)
def count_tokens(req: CountRequest):
    encoder = get_encoder()
    token_ids = encoder.encode(req.text)
    words = len(req.text.split())
    token_count = len(token_ids)

    price = MODEL_PRICES.get(req.model, MODEL_PRICES["gpt-4o-mini"])
    cost = token_count / 1_000_000 * price

    peek_tokens = []
    if req.peek > 0:
        peek_tokens = [encoder.decode([t]) for t in token_ids[: req.peek]]

    return CountResponse(
        chars=len(req.text),
        words=words,
        tokens=token_count,
        ratio=(token_count / words) if words else 0.0,
        cost=cost,
        model=req.model,
        peek_tokens=peek_tokens,
    )
