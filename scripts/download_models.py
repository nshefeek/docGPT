#! .venv/bin/python
import os
import requests
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.environ.get("MODEL_PATH", "./model_cache/")
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen2-VL-2B-Instruct-Q6_K.gguf")
LLM_MODEL_URL = os.environ.get("LLM_MODEL_URL", "https://huggingface.co/bartowski/Qwen2-VL-2B-Instruct-GGUF/resolve/main/Qwen2-VL-2B-Instruct-Q6_K.gguf")

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 KB

    with open(filename, "wb") as file, tqdm(
        desc=filename,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            progress_bar.update(size)


def ensure_model(model_name, url):
    model_path = f"{MODEL_PATH}{model_name}"
    if not os.path.exists(model_path):
        print(f"Downloading {model_name}...")
        download_file(url, model_path)
        print(f"{model_name} downloaded successfully.")
    else:
        print(f"{model_name} already exists. Skipping download.")


# LLM Model

def main():
    ensure_model(LLM_MODEL, LLM_MODEL_URL)
    print("All required models have been downloaded.")

if __name__ == "__main__":
    main()