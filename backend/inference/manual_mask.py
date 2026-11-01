"""Generate a single-frame mask from a user-supplied box without changing model outputs."""
import math
import threading
from pathlib import Path

_LOCK = threading.Lock()
_ROOT = Path(__file__).resolve().parents[2]


def segment_frame(video_path, frame_index, bbox):
    import cv2
    import numpy as np
    import torch
    from ultralytics import SAM

    if not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index < 0:
        raise ValueError('Invalid frame index')
    if len(bbox) != 4 or not all(math.isfinite(v) for v in bbox):
        raise ValueError('Box must contain four finite coordinates')
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ValueError('Video cannot be opened')
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_index >= count:
            raise ValueError('Frame is outside this video')
        x1, y1, x2, y2 = bbox
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError('Box must be inside the video frame')
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise ValueError('Video frame could not be decoded')
    finally:
        cap.release()

    weights = _ROOT / 'sam2.1_t.pt'
    if not weights.is_file():
        raise RuntimeError('SAM 2 model is not available')
    device = 'cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    with _LOCK:
        model = SAM(weights)
        with torch.inference_mode():
            result = model.predict(frame, bboxes=[list(bbox)], imgsz=1024, device=device, verbose=False)[0]
        polygons = [] if result.masks is None else result.masks.xy
        if not len(polygons) or len(polygons[0]) < 3:
            raise ValueError('No mask found for this box; try adjusting it')
        points = np.asarray(polygons[0], dtype=np.float32).reshape(-1, 1, 2)
        epsilon = max(.5, .001 * cv2.arcLength(points, True))
        polygon = cv2.approxPolyDP(points, epsilon, True).reshape(-1, 2)
        if len(polygon) < 3 or not np.isfinite(polygon).all():
            raise ValueError('The generated mask was invalid')
        return [[round(float(x), 1), round(float(y), 1)] for x, y in polygon]
