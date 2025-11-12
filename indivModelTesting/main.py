from huggingface_hub import snapshot_download
import os

# The model repository on Hugging Face
REPO_ID = "mlx-community/gemma-3-4b-it-4bit"

# The local directory where you want to save it
# We'll put it in its own subfolder to keep things organized
LOCAL_DIR = "./mlx_models/gemma3-4b-it-4bit"

# Create the directory if it doesn't exist
os.makedirs(LOCAL_DIR, exist_ok=True)

print(f"Downloading model '{REPO_ID}' to '{LOCAL_DIR}'...")

# Download all files from the repository to your local directory
snapshot_download(
    repo_id=REPO_ID,
    local_dir=LOCAL_DIR,
    local_dir_use_symlinks=False
)

print(f"Model successfully downloaded to: {LOCAL_DIR}")