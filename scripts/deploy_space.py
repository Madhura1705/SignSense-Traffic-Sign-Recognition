"""
Deploy SignSense to Hugging Face Spaces.
Uses cached authentication token.
"""
import glob
import os
import gradio as gr
from huggingface_hub import HfApi, create_repo, upload_folder, upload_file

def main():
    api = HfApi()
    user = api.whoami()["name"]
    repo_id = f"{user}/SignSense"
    print(f"Target Space: {repo_id}")

    # 1. Create Space if it doesn't exist
    try:
        url = create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            exist_ok=True
        )
        print(f"Space ready at: {url}")
    except Exception as e:
        print(f"create_repo notice: {e}")

    # 2. Create Space README with YAML front-matter
    gradio_ver = gr.__version__
    space_readme = f"""---
title: SignSense
emoji: 🚦
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: {gradio_ver}
app_file: app.py
pinned: false
---

# 🚦 SignSense: German Traffic Sign Recognition

Live deep learning traffic sign recognition on GTSRB (43 classes) in PyTorch.

- **Models:** Custom CNN (scratch), ResNet-18 (transfer learning), MobileNetV3-Small (edge deployment), Robust CNN (FGSM adversarial defense)
- **Features:** Grad-CAM explanations, model ensemble voting, live FGSM attack & defense, ONNX Runtime edge benchmarking
"""

    temp_readme = "temp_space_README.md"
    with open(temp_readme, "w", encoding="utf-8") as f:
        f.write(space_readme)

    try:
        api.upload_file(
            path_or_fileobj=temp_readme,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="space"
        )
        print("Uploaded Space README.md")

        # Upload app.py, ABOUT.md, requirements.txt
        for f in ["app.py", "ABOUT.md", "requirements.txt"]:
            api.upload_file(
                path_or_fileobj=f,
                path_in_repo=f,
                repo_id=repo_id,
                repo_type="space"
            )
            print(f"Uploaded {f}")

        # Upload folders: src, models, outputs, samples
        for folder in ["src", "models", "outputs", "samples"]:
            print(f"Uploading folder: {folder}...")
            api.upload_folder(
                folder_path=folder,
                path_in_repo=folder,
                repo_id=repo_id,
                repo_type="space",
                ignore_patterns=["*.onnx", "*.zip", ".gitkeep"]
            )
            print(f"Uploaded {folder}/ successfully.")

        print(f"\n🎉 Deployment complete!")
        print(f"Live Space URL: https://huggingface.co/spaces/{repo_id}")

    finally:
        if os.path.exists(temp_readme):
            os.remove(temp_readme)

if __name__ == "__main__":
    main()
