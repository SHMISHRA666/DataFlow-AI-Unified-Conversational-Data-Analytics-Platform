import os, requests

url = os.getenv("EMBED_API_URL")
model = os.getenv("EMBED_MODEL")

payload = {
    "model": model,
    "input": "This is a test sentence for embedding."
}

resp = requests.post(url, json=payload)
print(resp.json())