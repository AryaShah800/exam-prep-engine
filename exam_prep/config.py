"""Paths and settings. Everything can be overridden with environment variables or a .env file."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def set_root(root: Path) -> None:
    """Point every data path at <root>/data. Called once at import; tests call it with a temp folder."""
    g = globals()
    g["ROOT"] = Path(root)
    g["DATA"] = data = Path(root) / "data"
    g["RAW_DIR"] = data / "raw"                       # data/raw/<SUBJECT>/<SUBJECT>_<YYYY>_<MM>_<EXAM>.pdf
    g["EXTERNAL_DIR"] = data / "external"             # other universities: data/external/<SUBJECT>/<SRC>-<SUBJECT>_<YYYY>_<MM>_<EXAM>.pdf
    g["MANIFEST"] = data / "papers.csv"               # optional: overrides metadata parsed from file names
    g["SYLLABUS_DIR"] = data / "syllabus"             # data/syllabus/<SUBJECT>.txt or .pdf
    g["PAGES_DIR"] = data / "interim" / "pages"       # one <paper_id>.jsonl per paper
    g["PROCESSED_DIR"] = processed = data / "processed"
    g["QUESTIONS_FILE"] = processed / "questions.jsonl"
    g["TAGS_FILE"] = processed / "tags.jsonl"
    g["OVERRIDES_FILE"] = processed / "tag_overrides.jsonl"   # your hand corrections; always win over the model
    g["EXPORT_FILE"] = processed / "questions_tagged.jsonl"   # Phase B reads this
    g["TAXONOMY_DIR"] = data / "taxonomy"             # <SUBJECT>.json, editable by hand
    g["REVIEW_DIR"] = data / "review"
    g["CACHE_DIR"] = data / "cache" / "llm"         # embeddings are cached next to it in data/cache/embeddings/
    g["GOLDEN_DIR"] = data / "golden"                 # Phase C query sets


set_root(Path(os.getenv("EXAM_PREP_ROOT", Path(__file__).resolve().parent.parent)))

# Demo mode: the bundled synthetic sample bank, no LLM. Topic history counts questions whose source is PRIMARY_SOURCE.
OFFLINE = os.getenv("EXAM_PREP_OFFLINE", "").lower() in ("1", "true", "yes")
PRIMARY_SOURCE = os.getenv("EXAM_PREP_PRIMARY_SOURCE", "DDU")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
# Text work (splitting, tagging): a Flash-Lite model has the most generous free-tier limits.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
# Reading scanned pages and syllabus structuring: a stronger Flash model.
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-3.8-flash")
GEMINI_RPM = int(os.getenv("GEMINI_RPM", "10"))          # stay under your free-tier requests per minute
GEMINI_VISION_RPM = int(os.getenv("GEMINI_VISION_RPM", "4"))   # the Flash model's free tier is lower (5 RPM, 20/day)
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "8"))   # free-tier 503 "high demand" spikes can last minutes
GEMINI_TIMEOUT_S = int(os.getenv("GEMINI_TIMEOUT_S", "120"))   # per request; a stalled call is retried instead of hanging

# A page with fewer extractable characters than this is treated as scanned and sent to Gemini vision.
SCANNED_PAGE_MIN_CHARS = int(os.getenv("SCANNED_PAGE_MIN_CHARS", "60"))
OCR_DPI = int(os.getenv("OCR_DPI", "200"))
TAG_BATCH_SIZE = int(os.getenv("TAG_BATCH_SIZE", "8"))

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

# ---- Phase B: retrieval ----
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "gemini")          # gemini (API, ~1,000 texts/day free) or local (fastembed, offline)
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))           # truncated Matryoshka output; pgvector's HNSW index allows up to 2000 dims
GEMINI_EMBED_RPM = int(os.getenv("GEMINI_EMBED_RPM", "20"))
RRF_K = int(os.getenv("RRF_K", "60"))                    # Reciprocal Rank Fusion constant (Cormack et al. 2009)
RETRIEVE_CANDIDATES = int(os.getenv("RETRIEVE_CANDIDATES", "50"))   # each ranker's list length before fusion
# Score multiplier for other universities' questions so DDU questions win near-ties. RRF scores are flat
# (1/61, 1/62, ...), so 0.95 costs an external question about 3 places near the top; 0.8 would cost ~17. Tuned in Phase C.
EXTERNAL_WEIGHT = float(os.getenv("EXTERNAL_WEIGHT", "0.95"))
RETRIEVAL_BACKEND = os.getenv("RETRIEVAL_BACKEND", "memory")      # memory (numpy + BM25) or pg (Postgres + pgvector)
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/examprep")
ANSWER_MAX_ATTEMPTS = int(os.getenv("ANSWER_MAX_ATTEMPTS", "3"))   # answers whose citations fail validation are retried
# ---- Phase D slice: practice questions and rubric grading (exam_prep/intelligence.py) ----
INTELLIGENCE_MAX_ATTEMPTS = int(os.getenv("INTELLIGENCE_MAX_ATTEMPTS", "3"))   # generated questions / grades failing validation are retried

# ---- Local web app (exam_prep/web.py) ----
# Host headers the app accepts. Blocks DNS-rebinding pages from driving the local API (and the Gemini quota).
# Add your domain when deploying, e.g. WEB_ALLOWED_HOSTS=localhost,127.0.0.1,myspace.hf.space
WEB_ALLOWED_HOSTS = [h.strip() for h in os.getenv("WEB_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",") if h.strip()]

# ---- Phase C: evaluation ----
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///" + (ROOT / "mlflow.db").as_posix())   # `mlflow ui --backend-store-uri sqlite:///mlflow.db`
