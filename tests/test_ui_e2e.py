"""Browser test of the three demo flows (search, practice + marking, topic history). Skips without Playwright/Edge."""
import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
pytest.importorskip("playwright")
if not (Path(__file__).resolve().parents[1] / "demo/data/processed/questions_tagged.jsonl").exists():
    pytest.skip("demo bank missing", allow_module_level=True)
import demo_screenshots  # noqa: E402


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_three_flows_in_a_real_browser():
    port = free_port()
    server = demo_screenshots.start_server(port)
    try:
        try:
            out = demo_screenshots.run_flows(port)
        except Exception as error:   # no Edge/Chromium available
            if "Executable doesn't exist" in str(error) or "msedge" in str(error).lower() and "not found" in str(error).lower():
                pytest.skip("Edge not available")
            raise
    finally:
        server.terminate()
    assert out["search_hits"] == 5 and "deadlock" in out["search_top"].lower()
    assert out["grade_total"].endswith("/ 10") or "/" in out["grade_total"]
    assert out["history_rows"] == 8
