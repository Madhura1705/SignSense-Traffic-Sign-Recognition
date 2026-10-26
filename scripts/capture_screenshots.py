"""
Capture high-resolution screenshots of all 5 Gradio app tabs:
  - docs/screenshots/recognise.png
  - docs/screenshots/compare.png
  - docs/screenshots/attack.png
  - docs/screenshots/report.png
  - docs/screenshots/about.png
"""
import os
import subprocess
import time
import urllib.request
from playwright.sync_api import sync_playwright

def wait_for_server(url, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(1)
    return False

def main():
    os.makedirs("docs/screenshots", exist_ok=True)
    edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    if not os.path.exists(edge_path):
        edge_path = r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"

    port = 7867
    url = f"http://127.0.0.1:{port}"
    sample_img = os.path.abspath("samples/14_stop.png")

    print(f"Starting app.py on port {port}...")
    server = subprocess.Popen(
        [r".venv\Scripts\python.exe", "app.py", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    try:
        if not wait_for_server(url, timeout=40):
            print("Failed to connect to Gradio app server.")
            return

        print("Gradio app running. Launching Playwright browser...")
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=edge_path, headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 950})
            page.goto(url)
            page.wait_for_timeout(3000)

            # --- Tab 1: Recognise ---
            print("Capturing Tab 1: Recognise...")
            tab1 = page.locator("button[role='tab']:has-text('Recognise')")
            tab1.click()
            page.wait_for_timeout(1000)
            file_input = page.locator(".tabitem:visible input[type='file']").first
            if file_input.count() > 0:
                file_input.set_input_files(sample_img)
                page.wait_for_timeout(3500)
            page.screenshot(path="docs/screenshots/recognise.png", full_page=False)
            print("Saved docs/screenshots/recognise.png")

            # --- Tab 2: Compare models ---
            print("Capturing Tab 2: Compare models...")
            tab2 = page.locator("button[role='tab']:has-text('Compare models')")
            tab2.click()
            page.wait_for_timeout(1500)
            file_input2 = page.locator(".tabitem:visible input[type='file']").first
            if file_input2.count() > 0:
                file_input2.set_input_files(sample_img)
                page.wait_for_timeout(4500)
            page.screenshot(path="docs/screenshots/compare.png", full_page=False)
            print("Saved docs/screenshots/compare.png")

            # --- Tab 3: Fool the CNN ---
            print("Capturing Tab 3: Fool the CNN...")
            tab3 = page.locator("button[role='tab']:has-text('Fool the CNN')")
            tab3.click()
            page.wait_for_timeout(1500)
            file_input3 = page.locator(".tabitem:visible input[type='file']").first
            if file_input3.count() > 0:
                file_input3.set_input_files(sample_img)
                page.wait_for_timeout(1500)
            abtn = page.locator(".tabitem:visible button:has-text('Attack')")
            if abtn.count() > 0:
                abtn.click()
                page.wait_for_timeout(4000)
            page.screenshot(path="docs/screenshots/attack.png", full_page=False)
            print("Saved docs/screenshots/attack.png")

            # --- Tab 4: Model report ---
            print("Capturing Tab 4: Model report...")
            tab4 = page.locator("button[role='tab']:has-text('Model report')")
            tab4.click()
            page.wait_for_timeout(3000)
            page.screenshot(path="docs/screenshots/report.png", full_page=False)
            print("Saved docs/screenshots/report.png")

            # --- Tab 5: About ---
            print("Capturing Tab 5: About...")
            tab5 = page.locator("button[role='tab']:has-text('About')")
            tab5.click()
            page.wait_for_timeout(2000)
            page.screenshot(path="docs/screenshots/about.png", full_page=False)
            print("Saved docs/screenshots/about.png")

            browser.close()
    finally:
        print("Stopping app server...")
        server.terminate()
        server.wait()

    print("All 5 screenshots captured successfully!")

if __name__ == "__main__":
    main()
