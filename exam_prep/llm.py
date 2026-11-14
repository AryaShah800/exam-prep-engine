"""Gemini wrapper: structured JSON output, a disk cache, a requests-per-minute limiter and retries.

Every call is cached on disk by (model, prompt, schema, image bytes), so re-running a step never spends
free-tier quota twice. Tests swap in FakeLLM, which follows the same interface.
"""
import hashlib, json, time
from typing import Optional, Protocol, TypeVar
from pydantic import BaseModel
from . import config

M = TypeVar("M", bound=BaseModel)


class LLM(Protocol):
    model: str

    def generate_json(self, prompt: str, schema: type[M], images: Optional[list[bytes]] = None) -> M: ...


class LLMError(RuntimeError):
    pass


def _cache_key(model: str, prompt: str, schema: type[BaseModel], images: Optional[list[bytes]]) -> str:
    h = hashlib.sha256()
    for part in (model, schema.__name__, json.dumps(schema.model_json_schema(), sort_keys=True), prompt):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    for img in images or []:
        h.update(hashlib.sha256(img).digest())
    return h.hexdigest()


class GeminiLLM:
    """Calls the Gemini API through the google-genai SDK."""

    _last_call = 0.0

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None, rpm: int = config.GEMINI_RPM, use_cache: bool = True):
        from google import genai
        key = api_key or config.GEMINI_API_KEY
        if not key:
            raise LLMError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key from https://aistudio.google.com/apikey")
        from google.genai import types
        self.client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=config.GEMINI_TIMEOUT_S * 1000))
        self.model = model or config.GEMINI_MODEL
        self.min_interval = 60.0 / max(rpm, 1)
        self.use_cache = use_cache
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _throttle(self) -> None:
        wait = GeminiLLM._last_call + self.min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        GeminiLLM._last_call = time.monotonic()

    def generate_json(self, prompt: str, schema: type[M], images: Optional[list[bytes]] = None) -> M:
        from google.genai import types, errors
        key = _cache_key(self.model, prompt, schema, images)
        cache_file = config.CACHE_DIR / f"{key}.json"
        if self.use_cache and cache_file.exists():
            return schema.model_validate_json(cache_file.read_text(encoding="utf-8"))

        contents: list = [types.Part.from_bytes(data=img, mime_type="image/png") for img in images or []]
        contents.append(prompt)
        cfg = types.GenerateContentConfig(response_mime_type="application/json", response_schema=schema, temperature=0)

        last_err: Optional[Exception] = None
        for attempt in range(config.GEMINI_MAX_RETRIES):
            self._throttle()
            try:
                resp = self.client.models.generate_content(model=self.model, contents=contents, config=cfg)
                result = schema.model_validate_json(resp.text)
                cache_file.write_text(result.model_dump_json(), encoding="utf-8")
                return result
            except errors.APIError as e:
                last_err = e
                if is_daily_quota(e):
                    raise LLMError(f"Gemini daily free-tier quota used up for {self.model}; it resets at midnight Pacific time. {e}") from e
                if getattr(e, "code", None) not in (429, 500, 502, 503, 504):
                    raise LLMError(f"Gemini API error {getattr(e, 'code', '?')}: {e}") from e
            except ValueError as e:      # malformed JSON from the model: retry once more
                last_err = e
            except Exception as e:       # dropped connection / timeout (httpx errors, WinError 10054): retry
                if not _is_network_error(e):
                    raise
                last_err = e
            backoff = min(60, 2 ** attempt * 4)
            print(f"  Gemini call failed ({last_err.__class__.__name__}); retrying in {backoff}s")
            time.sleep(backoff)
        raise LLMError(f"Gemini call failed after {config.GEMINI_MAX_RETRIES} attempts: {last_err}")


def is_daily_quota(e: Exception) -> bool:
    """A per-day quota 429 will not clear in seconds, and every retry spends another request: stop at once."""
    return getattr(e, "code", None) == 429 and "PerDay" in str(e)


def _is_network_error(e: Exception) -> bool:
    mod = type(e).__module__ or ""
    return isinstance(e, (ConnectionError, TimeoutError)) or mod.startswith(("httpx", "httpcore"))


def load_prompt(name: str) -> tuple[str, str]:
    """Returns (template, version). Prompt files are named <name>_v<N>.txt; the highest N wins."""
    files = sorted(config.PROMPTS_DIR.glob(f"{name}_v*.txt"), key=lambda p: int(p.stem.rsplit("_v", 1)[1]))
    if not files:
        raise FileNotFoundError(f"No prompt file for {name!r} in {config.PROMPTS_DIR}")
    latest = files[-1]
    return latest.read_text(encoding="utf-8"), latest.stem
