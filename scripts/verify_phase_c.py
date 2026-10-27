import asyncio
import os
import subprocess
import sys
import time
import urllib.request
from playwright.async_api import async_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(ROOT, "docs", "screenshots"), exist_ok=True)

async def main():
    print("1. Starting server.py...")
    proc = subprocess.Popen(
        [sys.executable, "server.py", "--port", "7860", "--no-browser"],
        cwd=ROOT,
    )

    # Poll server health
    print("   Waiting for server...")
    server_ready = False
    for i in range(40):
        try:
            with urllib.request.urlopen("http://127.0.0.1:7860/api/health", timeout=1) as resp:
                if resp.status == 200:
                    server_ready = True
                    print(f"   Server ready after {i*0.5:.1f}s!")
                    break
        except Exception:
            time.sleep(0.5)

    if not server_ready:
        proc.terminate()
        raise RuntimeError("Server failed to respond within 20s.")

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(channel="msedge", headless=True)

            # ==========================================
            # DESKTOP (1440 x 900)
            # ==========================================
            print("2. Verifying Home Page (Desktop 1440x900)...")
            context_d = await browser.new_context(viewport={"width": 1440, "height": 900})
            page_d = await context_d.new_page()

            await page_d.goto("http://127.0.0.1:7860/#/", wait_until="networkidle")
            await page_d.wait_for_selector("#ticker-pred-name", timeout=10000)
            await page_d.wait_for_selector("#roundel-acc", timeout=10000)
            await page_d.wait_for_selector(".module-card", timeout=10000)

            # Wait a moment for dynamic metrics to populate
            await page_d.wait_for_timeout(1500)

            shot_home_d = os.path.join(ROOT, "docs", "screenshots", "v2_home_desktop.png")
            await page_d.screenshot(path=shot_home_d, full_page=True)
            print(f"   Captured: {shot_home_d}")

            print("3. Verifying Recognise Page (Desktop 1440x900)...")
            await page_d.goto("http://127.0.0.1:7860/#/recognise", wait_until="networkidle")
            await page_d.wait_for_selector(".sample-thumb", timeout=10000)

            # Click Stop sample sign
            stop_thumb = page_d.locator("img.sample-thumb[alt*='stop' i], img.sample-thumb[title*='stop' i]").first
            await stop_thumb.click()
            print("   Clicked Stop sample sign...")

            # Wait for prediction result card
            await page_d.wait_for_selector(".prediction-class-name", timeout=10000)
            pred_title = await page_d.text_content(".prediction-class-name")
            print(f"   Predicted class: {pred_title.strip()}")
            assert "Stop" in pred_title, f"Expected Stop, got {pred_title}"

            await page_d.wait_for_selector(".cam-viewer", timeout=10000)
            await page_d.wait_for_selector(".top5-item", timeout=10000)

            shot_rec_d = os.path.join(ROOT, "docs", "screenshots", "v2_recognise_desktop.png")
            await page_d.screenshot(path=shot_rec_d, full_page=True)
            print(f"   Captured: {shot_rec_d}")

            # ==========================================
            # MOBILE (390 x 844)
            # ==========================================
            print("4. Verifying Home Page (Mobile 390x844)...")
            context_m = await browser.new_context(viewport={"width": 390, "height": 844})
            page_m = await context_m.new_page()

            await page_m.goto("http://127.0.0.1:7860/#/", wait_until="networkidle")
            await page_m.wait_for_selector(".roundel", timeout=10000)
            await page_m.wait_for_timeout(1000)

            shot_home_m = os.path.join(ROOT, "docs", "screenshots", "v2_home_mobile.png")
            await page_m.screenshot(path=shot_home_m, full_page=True)
            print(f"   Captured: {shot_home_m}")

            print("5. Verifying Recognise Page (Mobile 390x844)...")
            await page_m.goto("http://127.0.0.1:7860/#/recognise", wait_until="networkidle")
            await page_m.wait_for_selector(".sample-thumb", timeout=10000)

            stop_thumb_m = page_m.locator("img.sample-thumb[alt*='stop' i], img.sample-thumb[title*='stop' i]").first
            await stop_thumb_m.click()
            await page_m.wait_for_selector(".prediction-class-name", timeout=10000)
            await page_m.wait_for_timeout(1000)

            shot_rec_m = os.path.join(ROOT, "docs", "screenshots", "v2_recognise_mobile.png")
            await page_m.screenshot(path=shot_rec_m, full_page=True)
            print(f"   Captured: {shot_rec_m}")

            await browser.close()
            print("6. All Phase C screenshots verified and captured successfully!")
    finally:
        print("7. Stopping server...")
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()
        print("Server stopped.")

if __name__ == "__main__":
    asyncio.run(main())
