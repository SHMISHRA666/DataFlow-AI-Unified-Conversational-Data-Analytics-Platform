import requests

url = "http://localhost:11434/api/generate"
payload = {
    "model": "gemma2:9b",
    "prompt": "Write a short funny sentence about AI and coffee."
}

resp = requests.post(url, json=payload, stream=True)
for line in resp.iter_lines():
    if line:
        print(line.decode())