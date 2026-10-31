"""Download public model weights; no user media is transmitted."""
from pathlib import Path
from urllib.request import urlretrieve

target = Path(__file__).parent / 'inference/models/hand_landmarker.task'
target.parent.mkdir(parents=True, exist_ok=True)
if not target.exists():
    urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task', target)
print(target)
