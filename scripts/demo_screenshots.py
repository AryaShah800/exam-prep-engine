"""Drive the demo UI with Playwright (system Edge) and save screenshots to docs/.

    python scripts/demo_screenshots.py
Also used by tests/test_ui_e2e.py via run_flows(). Needs: pip install playwright, Microsoft Edge installed.
"""
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANSWER = ("A deadlock needs four conditions: mutual exclusion, hold and wait, no preemption and circular wait. "
          "Avoidance keeps the system in a safe state, and the Banker algorithm checks the available resources "
          "against the maximum need of every process before granting a request. Detection uses a wait-for graph and "
          "recovery can terminate a process or preempt resources.")


def start_server(port):
    proc = subprocess.Popen([sys.executable, "-m", "exam_prep", "demo", "--port", str(port)], cwd=ROOT)
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return proc
        except OSError:
            time.sleep(1)
    proc.terminate()
    raise RuntimeError("demo server did not start")


def run_flows(port, shots=None):
    from playwright.sync_api import sync_playwright
    out = {}
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context("D:/work/_cache/edge-aiml-ep", channel="msedge", headless=True,
                                                   viewport={"width": 1100, "height": 900})
        page = ctx.new_page()
        page.goto(f"http://127.0.0.1:{port}/")
        page.locator("#banner:not([hidden])").wait_for()
        # flow 1: search
        page.fill("#query", "deadlock conditions")
        page.select_option("#subject", "OS")
        page.get_by_role("button", name="Search papers").click()
        page.locator("#results .q").first.wait_for()
        out["search_hits"] = page.locator("#results .q").count()
        out["search_top"] = page.locator("#results .q p").first.inner_text()
        if shots:
            page.screenshot(path=str(shots / "ui-search.png"))
        # flow 2: practice and marking
        page.click("#t-practice")
        page.fill("#ptopic", "deadlock")
        page.select_option("#psubject", "OS")
        page.get_by_role("button", name="New practice question").click()
        page.locator("#presults .q").wait_for()
        page.fill("#panswer", ANSWER)
        page.get_by_role("button", name="Mark my answer").click()
        page.locator("#gresults .total").wait_for()
        out["grade_total"] = page.locator("#gresults .total").inner_text()
        if shots:
            page.screenshot(path=str(shots / "ui-practice.png"), full_page=True)
        # flow 3: topic history
        page.click("#t-history")
        page.select_option("#hsubject", "OS")
        page.get_by_role("button", name="Show topic history").click()
        page.locator("#hresults .hist tr").first.wait_for()
        out["history_rows"] = page.locator("#hresults .hist tr").count()
        if shots:
            page.screenshot(path=str(shots / "ui-history.png"), full_page=True)
        ctx.close()
    return out


if __name__ == "__main__":
    server = start_server(8012)
    try:
        print(run_flows(8012, ROOT / "docs"))
    finally:
        server.terminate()
