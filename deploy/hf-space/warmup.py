"""Build-time warm-up for the hosted demo: downloads the local embedding model and embeds the sample bank,
so a visitor's first hybrid or semantic search does not wait for a download. Run with WEB_ALLOWED_HOSTS=testserver."""
from pathlib import Path
from exam_prep import config

config.set_root(Path("demo").resolve())
config.OFFLINE = True
config.PRIMARY_SOURCE = "SAMPLE"
from fastapi.testclient import TestClient  # noqa: E402
from exam_prep.web import app  # noqa: E402

client = TestClient(app)
for mode in ("bm25", "hybrid"):
    r = client.post("/search", json={"query": "deadlock prevention", "mode": mode})
    r.raise_for_status()
    print(mode, "->", len(r.json().get("hits", r.json())), "hits")
