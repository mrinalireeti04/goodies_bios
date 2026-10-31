"""List the Gemini models that support generateContent for the key in GEMINI_API_KEY (the key is never printed).

Usage: python dev/gemini_models.py
"""
import json
import os
import sys
import urllib.error
import urllib.request

URL = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200"


def list_models(key: str) -> list[dict]:
    models, token = [], ""
    while True:
        req = urllib.request.Request(URL + (f"&pageToken={token}" if token else ""), headers={"x-goog-api-key": key})
        with urllib.request.urlopen(req, timeout=30) as resp:
            page = json.loads(resp.read())
        models += page.get("models", [])
        token = page.get("nextPageToken", "")
        if not token:
            return [m for m in models if "generateContent" in m.get("supportedGenerationMethods", [])]


if __name__ == "__main__":
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY is not set in this process")
    try:
        found = list_models(key)
    except urllib.error.HTTPError as e:
        sys.exit(f"ListModels failed: HTTP {e.code}")
    for m in sorted(found, key=lambda m: m["name"]):
        print(m["name"].removeprefix("models/"), "|", m.get("displayName", ""), "| in", m.get("inputTokenLimit", "?"))
    print(f"{len(found)} models support generateContent")
