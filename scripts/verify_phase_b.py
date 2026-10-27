import asyncio
import os
import subprocess
import sys
import time
from playwright.async_api import async_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(ROOT, "docs", "screenshots"), exist_ok=True)

async def main():
    # 1. Start server.py process
    print("1. Starting server.py...")
    proc = subprocess.Popen(
        [sys.executable, "server.py", "--port", "7860", "--no-browser"],
        cwd=ROOT,
    )
    
    # Poll until server is ready
    print("   Waiting for server at http://127.0.0.1:7860/api/health...")
    import urllib.request
    server_ready = False
    for i in range(40):
        try:
            with urllib.request.urlopen("http://127.0.0.1:7860/api/health", timeout=1) as resp:
                if resp.status == 200:
                    server_ready = True
                    print(f"   Server responded with 200 OK after {i*0.5:.1f}s!")
                    break
        except Exception:
            time.sleep(0.5)

    if not server_ready:
        raise RuntimeError("Server failed to respond within 20 seconds.")

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(channel="msedge", headless=True)

            # --- Desktop Viewport (1440 x 900) ---
            print("2. Testing Desktop Viewport (1440x900)...")
            context_desktop = await browser.new_context(viewport={"width": 1440, "height": 900})
            page_desktop = await context_desktop.new_page()

            await page_desktop.goto("http://127.0.0.1:7860/#/", wait_until="networkidle")
            await page_desktop.wait_for_selector("#status-pill:not(.offline)", timeout=10000)
            status_text = await page_desktop.text_content("#status-text")
            print(f"   Status pill text: {status_text.strip()}")
            assert "4 models ready" in status_text, f"Expected 4 models ready, got {status_text}"

            desktop_shot = os.path.join(ROOT, "docs", "screenshots", "v2_shell_desktop.png")
            await page_desktop.screenshot(path=desktop_shot, full_page=False)
            print(f"   Captured: {desktop_shot}")

            # Test Navigation & Back/Forward
            print("   Testing hash navigation...")
            await page_desktop.click("#nav-recognise")
            await page_desktop.wait_for_timeout(300)
            assert page_desktop.url.endswith("#/recognise")

            await page_desktop.click("#nav-compare")
            await page_desktop.wait_for_timeout(300)
            assert page_desktop.url.endswith("#/compare")

            print("   Testing browser back and forward...")
            await page_desktop.go_back()
            await page_desktop.wait_for_timeout(300)
            assert page_desktop.url.endswith("#/recognise")

            await page_desktop.go_forward()
            await page_desktop.wait_for_timeout(300)
            assert page_desktop.url.endswith("#/compare")

            # --- Mobile Viewport (390 x 844) ---
            print("3. Testing Mobile Viewport (390x844)...")
            context_mobile = await browser.new_context(viewport={"width": 390, "height": 844})
            page_mobile = await context_mobile.new_page()

            await page_mobile.goto("http://127.0.0.1:7860/#/", wait_until="networkidle")
            await page_mobile.wait_for_selector("#status-pill:not(.offline)", timeout=10000)

            # Open hamburger menu
            hamburger = page_mobile.locator("#hamburger-btn")
            assert await hamburger.is_visible(), "Hamburger button should be visible on mobile"
            await hamburger.click()
            await page_mobile.wait_for_timeout(300)

            drawer = page_mobile.locator("#mobile-drawer")
            assert await drawer.is_visible(), "Mobile drawer should be visible after clicking hamburger"

            mobile_shot = os.path.join(ROOT, "docs", "screenshots", "v2_shell_mobile.png")
            await page_mobile.screenshot(path=mobile_shot, full_page=False)
            print(f"   Captured: {mobile_shot}")

            await browser.close()
            print("4. Verification & Screenshots completed successfully!")
    finally:
        print("5. Stopping server...")
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()
        print("Server stopped.")

if __name__ == "__main__":
    asyncio.run(main())
