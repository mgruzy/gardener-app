"""Shared plant RAG retriever.

Used by both the General Garden Bot and the Layout Optimizer.
Loads the index built by build/plant_rag/main.py lazily on first call.

ASSUMPTIONS:
  - Embeddings are L2-normalised float32, shape (N, 1536)
  - Cosine similarity = dot product (embeddings already normalised)
  - Index lives at data/plant_rag/embeddings.npy + docs.json
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI


# ── Parameters ────────────────────────────────────────────────────────────────

_RAG_DIR      = Path(__file__).parents[2] / "data" / "plant_rag"
_EMB_PATH     = _RAG_DIR / "embeddings.npy"
_DOCS_PATH    = _RAG_DIR / "docs.json"

EMBED_MODEL   = "text-embedding-3-small"
DEFAULT_TOP_K = 4


# ── Module-level cache ────────────────────────────────────────────────────────

_embeddings: np.ndarray | None = None
_docs: list[dict] | None = None
_client: OpenAI | None = None


# ── Private helpers ───────────────────────────────────────────────────────────

def _load() -> None:
    """Load index files into memory on first call."""
    global _embeddings, _docs, _client
    if _embeddings is not None:
        return
    if not _EMB_PATH.exists() or not _DOCS_PATH.exists():
        raise FileNotFoundError(
            f"RAG index not found at {_RAG_DIR}. "
            "Run build/plant_rag/main.py first."
        )
    _embeddings = np.load(str(_EMB_PATH))
    with open(_DOCS_PATH) as f:
        _docs = json.load(f)
    _client = OpenAI()


# ── Public API ────────────────────────────────────────────────────────────────

def is_ready() -> bool:
    """Return True if the index files exist on disk."""
    return _EMB_PATH.exists() and _DOCS_PATH.exists()


def search(query: str, k: int = DEFAULT_TOP_K) -> list[str]:
    """
    Return the top-k most relevant document texts for the query.

    Args:
        query: Natural-language question or keyword string.
        k:     Number of results to return.

    Returns:
        List of document text strings, ranked by cosine similarity.
    """
    _load()
    resp  = _client.embeddings.create(input=[query], model=EMBED_MODEL)
    q_vec = np.array(resp.data[0].embedding, dtype=np.float32)
    q_vec /= np.maximum(np.linalg.norm(q_vec), 1e-9)

    scores  = _embeddings @ q_vec            # (N,) cosine similarities
    top_idx = np.argsort(scores)[-k:][::-1]  # highest first
    return [_docs[i]["text"] for i in top_idx]


def search_with_scores(query: str, k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
    """
    Return the top-k results with their similarity scores and source metadata.

    Args:
        query: Natural-language question or keyword string.
        k:     Number of results to return.

    Returns:
        List of dicts with keys: text, source, score.
    """
    _load()
    resp  = _client.embeddings.create(input=[query], model=EMBED_MODEL)
    q_vec = np.array(resp.data[0].embedding, dtype=np.float32)
    q_vec /= np.maximum(np.linalg.norm(q_vec), 1e-9)

    scores  = _embeddings @ q_vec
    top_idx = np.argsort(scores)[-k:][::-1]
    return [
        {
            "text":   _docs[i]["text"],
            "source": _docs[i].get("source", "unknown"),
            "score":  float(scores[i]),
        }
        for i in top_idx
    ]
