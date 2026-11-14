"""Gemini text embeddings with a disk cache.

Documents and queries use different task types (RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY), as Gemini recommends.
Every vector is cached on disk by (model, dimensions, task, text), so rebuilding the index or re-running an
evaluation never spends free-tier quota twice. Vectors are L2-normalised, so a dot product is cosine similarity
(Gemini only normalises the full 3072-dim output; truncated vectors must be normalised by the caller).
"""
import hashlib, json, time
from pathlib import Path
from typing import Literal, Optional, Protocol
import numpy as np
from . import config
from .llm import LLMError, _is_network_error, is_daily_quota

Task = Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]


class Embedder(Protocol):
    model: str
    dim: int

    def embed(self, texts: list[str], task: Task) -> np.ndarray: ...


def normalise(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


class EmbeddingCache:
    """Append-only JSONL file: one {"k": sha256, "v": [...]} row per embedded text."""

    def __init__(self, model: str, dim: int):
        safe = model.replace("/", "_")
        self.path: Path = config.CACHE_DIR.parent / "embeddings" / f"{safe}-{dim}.jsonl"
        self.vectors: dict[str, list[float]] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        self.vectors[row["k"]] = row["v"]

    @staticmethod
    def key(task: str, text: str) -> str:
        return hashlib.sha256(f"{task}\x00{text}".encode("utf-8")).hexdigest()

    def add(self, items: dict[str, list[float]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            for k, v in items.items():
                f.write(json.dumps({"k": k, "v": [round(x, 6) for x in v]}) + "\n")
        self.vectors.update(items)


class CachedEmbedder:
    """Shared caching/batching logic; subclasses implement _embed_batch."""

    batch_size = 100

    def __init__(self, model: str, dim: int):
        self.model, self.dim = model, dim
        self.cache = EmbeddingCache(model, dim)
        self.api_calls = 0

    def _embed_batch(self, texts: list[str], task: Task) -> list[list[float]]:
        raise NotImplementedError

    def embed(self, texts: list[str], task: Task) -> np.ndarray:
        keys = [EmbeddingCache.key(task, t) for t in texts]
        missing = list(dict.fromkeys((k, t) for k, t in zip(keys, texts) if k not in self.cache.vectors))
        for i in range(0, len(missing), self.batch_size):
            chunk = missing[i:i + self.batch_size]
            vecs = self._embed_batch([t for _, t in chunk], task)
            if len(vecs) != len(chunk):
                raise LLMError(f"embedding model returned {len(vecs)} vectors for {len(chunk)} texts")
            self.cache.add({k: v for (k, _), v in zip(chunk, vecs)})
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return normalise(np.array([self.cache.vectors[k] for k in keys]))


class GeminiEmbedder(CachedEmbedder):
    """gemini-embedding-001 accepts up to 100 texts per call. (gemini-embedding-2 merges a list of texts into
    ONE vector, so it would need one call per text; that is why 001 is the default.)"""

    _last_call = 0.0

    def __init__(self, model: Optional[str] = None, dim: Optional[int] = None, api_key: Optional[str] = None):
        super().__init__(model or config.GEMINI_EMBED_MODEL, dim or config.EMBED_DIM)
        from google import genai
        from google.genai import types
        key = api_key or config.GEMINI_API_KEY
        if not key:
            raise LLMError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
        self.client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=config.GEMINI_TIMEOUT_S * 1000))
        self.min_interval = 60.0 / max(config.GEMINI_EMBED_RPM, 1)

    def _embed_batch(self, texts: list[str], task: Task) -> list[list[float]]:
        from google.genai import types, errors
        cfg = types.EmbedContentConfig(task_type=task, output_dimensionality=self.dim)
        last_err: Optional[Exception] = None
        for attempt in range(config.GEMINI_MAX_RETRIES):
            wait = GeminiEmbedder._last_call + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            GeminiEmbedder._last_call = time.monotonic()
            try:
                resp = self.client.models.embed_content(model=self.model, contents=texts, config=cfg)
                self.api_calls += 1
                return [list(e.values) for e in resp.embeddings]
            except errors.APIError as e:
                last_err = e
                if is_daily_quota(e):
                    raise LLMError(f"Gemini daily embedding quota used up for {self.model} (the free tier counts every text, about 1,000 a day); "
                                   "it resets at midnight Pacific time. Cached vectors still work.") from e
                if getattr(e, "code", None) not in (429, 500, 502, 503, 504):
                    raise LLMError(f"Gemini embedding error {getattr(e, 'code', '?')}: {e}") from e
            except Exception as e:
                if not _is_network_error(e):
                    raise
                last_err = e
            backoff = min(60, 2 ** attempt * 4)
            print(f"  embedding call failed ({last_err.__class__.__name__}); retrying in {backoff}s")
            time.sleep(backoff)
        raise LLMError(f"embedding failed after {config.GEMINI_MAX_RETRIES} attempts: {last_err}")


class LocalEmbedder(CachedEmbedder):
    """Free, offline embeddings with fastembed (ONNX Runtime, no PyTorch). Default BAAI/bge-base-en-v1.5: 768 dims like
    the Gemini setup, so the pgvector schema is unchanged. No API quota, so the app's per-query embedding never runs out.
    bge models expect an instruction prefix on queries; fastembed's query_embed adds it."""

    batch_size = 64

    def __init__(self, model: Optional[str] = None):
        from fastembed import TextEmbedding
        name = model or config.LOCAL_EMBED_MODEL
        self._model = TextEmbedding(model_name=name, cache_dir=str(config.CACHE_DIR.parent / "fastembed"))
        dim = len(next(iter(self._model.embed(["dimension probe"]))))
        super().__init__(name, dim)

    def _embed_batch(self, texts: list[str], task: Task) -> list[list[float]]:
        self.api_calls += 1
        it = self._model.query_embed(texts) if task == "RETRIEVAL_QUERY" else self._model.passage_embed(texts)
        return [list(map(float, v)) for v in it]


def make_embedder(provider: Optional[str] = None) -> CachedEmbedder:
    """EMBED_PROVIDER=gemini (default) or local. Each provider has its own cache file, so switching never mixes vectors."""
    provider = (provider or config.EMBED_PROVIDER).lower()
    if provider == "local":
        return LocalEmbedder()
    if provider == "gemini":
        return GeminiEmbedder()
    raise ValueError(f"unknown embedding provider {provider!r} (use gemini or local)")
