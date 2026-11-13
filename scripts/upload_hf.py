#!/usr/bin/env python3
"""Upload the trained LoRA adapter to Hugging Face Hub."""
import os
import sys
from huggingface_hub import HfApi

REPO_ID = "nikhilj007/ai-assign-lora14b-r1"
MODEL_DIR = "/scratch/r25cs0001/ai-assign/models/lora14b-r1"

def main():
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    
    print(f"Checking identity on Hugging Face...")
    user = api.whoami()
    print(f"Logged in as: {user.get('name')}")
    
    print(f"Creating repository: {REPO_ID} (if not already existing)...")
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
    
    print(f"Uploading files from {MODEL_DIR} to {REPO_ID}...")
    api.upload_folder(
        folder_path=MODEL_DIR,
        repo_id=REPO_ID,
        repo_type="model",
    )
    print(f"\nUpload complete! Model is live at: https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    main()
