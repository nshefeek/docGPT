import os
import requests
from tqdm import tqdm

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB

    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            progress_bar.update(size)

def ensure_model(model_name, url):
    model_path = f"/app/data/{model_name}"
    if not os.path.exists(model_path):
        print(f"Downloading {model_name}...")
        download_file(url, model_path)
        print(f"{model_name} downloaded successfully.")
    else:
        print(f"{model_name} already exists. Skipping download.")

# LLM Model
llm_model = os.getenv('LLM_MODEL', 'mistral-7b-openorca.gguf2.Q4_0.gguf')
llm_url = f"https://gpt4all.io/models/{llm_model}"

# Embedding Model (assuming it's a sentence transformer model)
embedding_model = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
embedding_url = f"https://huggingface.co/sentence-transformers/{embedding_model}/resolve/main/pytorch_model.bin"

ensure_model(llm_model, llm_url)
ensure_model(f"{embedding_model}.bin", embedding_url)

print("All required models have been downloaded.")