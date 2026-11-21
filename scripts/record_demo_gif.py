"""Record docs/demo.gif: search -> practice question -> marked answer -> topic history, in system Edge.

    python scripts/record_demo_gif.py
Frames are viewport screenshots stitched with Pillow (no video tooling). Needs: pip install playwright pillow, Microsoft Edge.
"""
import io
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_screenshots import ANSWER, ROOT, start_server  # noqa: E402

WIDTH = 820


def main(port=8013):
    from playwright.sync_api import sync_playwright
    frames = []

    def snap(page, seconds):
        image = Image.open(io.BytesIO(page.screenshot())).convert("RGB")
        image = image.resize((WIDTH, round(image.height * WIDTH / image.width)), Image.LANCZOS)
        frames.append((image.quantize(colors=96, method=Image.MEDIANCUT, dither=Image.NONE), int(seconds * 1000)))

    server = start_server(port)
    try:
        with sync_playwright() as p:
            ctx = p.chromium.launch_persistent_context("D:/work/_cache/edge-aiml-ep", channel="msedge", headless=True,
                                                       viewport={"width": 1100, "height": 760})
            page = ctx.new_page()
            page.goto(f"http://127.0.0.1:{port}/")
            page.locator("#banner:not([hidden])").wait_for()
            snap(page, 1.5)
            page.select_option("#subject", "OS")
            page.type("#query", "deadlock conditions", delay=40)
            snap(page, 1.2)
            page.get_by_role("button", name="Search papers").click()
            page.locator("#results .q").first.wait_for()
            snap(page, 3)
            page.click("#t-practice")
            page.select_option("#psubject", "OS")
            page.fill("#ptopic", "deadlock")
            snap(page, 1)
            page.get_by_role("button", name="New practice question").click()
            page.locator("#presults .q").wait_for()
            snap(page, 2.5)
            page.fill("#panswer", ANSWER)
            snap(page, 2)
            page.get_by_role("button", name="Mark my answer").click()
            page.locator("#gresults .total").wait_for()
            page.locator("#gresults .total").scroll_into_view_if_needed()
            snap(page, 3.5)
            page.click("#t-history")
            page.select_option("#hsubject", "OS")
            page.get_by_role("button", name="Show topic history").click()
            page.locator("#hresults .hist tr").first.wait_for()
            snap(page, 3)
            ctx.close()
    finally:
        server.terminate()
    out = ROOT / "docs" / "demo.gif"
    frames[0][0].save(out, save_all=True, append_images=[f for f, _ in frames[1:]], duration=[d for _, d in frames],
                      loop=0, optimize=True)
    print(out, out.stat().st_size // 1024, "KB", len(frames), "frames")


if __name__ == "__main__":
    main()
