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
            context_d = await browser.new_context(viewport={"width": 1440, "height": 900})
            page_d = await context_d.new_page()

            # 1. Compare Page (Desktop)
            print("2. Verifying Compare Page (Desktop 1440x900)...")
            await page_d.goto("http://127.0.0.1:7860/#/compare", wait_until="networkidle")
            await page_d.wait_for_selector(".sample-thumb", timeout=10000)

            # Click a sample (Stop sign)
            sample_btn = page_d.locator("img.sample-thumb[alt*='stop' i], img.sample-thumb[title*='stop' i]").first
            await sample_btn.click()
            print("   Clicked sample sign for comparison...")

            # Wait for ensemble banner and 4 model cards
            await page_d.wait_for_selector(".ensemble-banner", timeout=10000)
            await page_d.wait_for_selector(".compare-card", timeout=10000)
            await page_d.wait_for_selector("canvas", timeout=10000)
            await page_d.wait_for_timeout(1000)

            banner_text = await page_d.text_content(".ensemble-banner")
            safe_banner = banner_text.encode("ascii", "replace").decode("ascii")
            print(f"   Ensemble result: {safe_banner.strip()}")
            assert "agree: Stop" in banner_text, f"Expected consensus on Stop, got {banner_text}"

            shot_comp_d = os.path.join(ROOT, "docs", "screenshots", "v2_compare_desktop.png")
            await page_d.screenshot(path=shot_comp_d, full_page=True)
            print(f"   Captured: {shot_comp_d}")

            # 2. Attack Lab Page (Desktop)
            print("3. Verifying Attack Lab Page (Desktop 1440x900)...")
            await page_d.goto("http://127.0.0.1:7860/#/attack", wait_until="networkidle")
            await page_d.wait_for_selector(".preset-btn", timeout=10000)

            # Click preset "Speed 50 at ε 0.02"
            preset_btn = page_d.locator(".preset-btn").first
            await preset_btn.click()
            print("   Clicked demonstration preset...")

            # Wait for verdict and attack triplet
            await page_d.wait_for_selector(".verdict-banner", timeout=10000)
            await page_d.wait_for_selector(".attack-triplet", timeout=10000)
            await page_d.wait_for_selector(".transfer-table", timeout=10000)

            verdict_text = await page_d.text_content(".verdict-banner")
            safe_verdict = verdict_text.encode("ascii", "replace").decode("ascii")
            print(f"   Attack verdict: {safe_verdict.strip()}")

            # Test Robustness Sweep
            print("   Testing Robustness Sweep...")
            sweep_btn = page_d.locator("#run-sweep-btn")
            await sweep_btn.click()
            await page_d.wait_for_selector("#sweep-chart-canvas", timeout=20000)
            await page_d.wait_for_timeout(1500)

            shot_att_d = os.path.join(ROOT, "docs", "screenshots", "v2_attack_desktop.png")
            await page_d.screenshot(path=shot_att_d, full_page=True)
            print(f"   Captured: {shot_att_d}")

            # ==========================================
            # MOBILE (390 x 844)
            # ==========================================
            context_m = await browser.new_context(viewport={"width": 390, "height": 844})
            page_m = await context_m.new_page()

            # 1. Compare Page (Mobile)
            print("4. Verifying Compare Page (Mobile 390x844)...")
            await page_m.goto("http://127.0.0.1:7860/#/compare", wait_until="networkidle")
            await page_m.wait_for_selector(".sample-thumb", timeout=10000)
            sample_m = page_m.locator("img.sample-thumb[alt*='stop' i], img.sample-thumb[title*='stop' i]").first
            await sample_m.click()
            await page_m.wait_for_selector(".ensemble-banner", timeout=10000)
            await page_m.wait_for_timeout(1000)

            shot_comp_m = os.path.join(ROOT, "docs", "screenshots", "v2_compare_mobile.png")
            await page_m.screenshot(path=shot_comp_m, full_page=True)
            print(f"   Captured: {shot_comp_m}")

            # 2. Attack Lab Page (Mobile)
            print("5. Verifying Attack Lab Page (Mobile 390x844)...")
            await page_m.goto("http://127.0.0.1:7860/#/attack", wait_until="networkidle")
            await page_m.wait_for_selector(".preset-btn", timeout=10000)
            preset_m = page_m.locator(".preset-btn").first
            await preset_m.click()
            await page_m.wait_for_selector(".verdict-banner", timeout=10000)
            await page_m.wait_for_timeout(1000)

            shot_att_m = os.path.join(ROOT, "docs", "screenshots", "v2_attack_mobile.png")
            await page_m.screenshot(path=shot_att_m, full_page=True)
            print(f"   Captured: {shot_att_m}")

            await browser.close()
            print("6. All Phase D screenshots verified and captured successfully!")
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
