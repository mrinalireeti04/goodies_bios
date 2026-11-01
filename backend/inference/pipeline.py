"""Local hybrid inference. Raw measurements are retained; only bounded gaps are filled."""
import copy
import json
import os
import threading
from pathlib import Path

import cv2
import numpy as np
from . import hand_detect as mp
from .labels import correct_hand_sides
from .rtm_hand import RTMHandEngine, iou

INFERENCE_LOCK = threading.Lock()
MODEL = Path(__file__).parent / 'models' / 'hand_landmarker.task'
MEDIAPIPE_FALLBACK_THRESHOLD = 0.8


def needs_rtm_fallback(hands, threshold=MEDIAPIPE_FALLBACK_THRESHOLD):
    """Use handedness confidence as a proxy; it is not landmark accuracy."""
    return not hands or any(
        hand.get('side') not in ('left', 'right')
        or hand.get('handedness_score') is None
        or not np.isfinite(hand['handedness_score'])
        or hand['handedness_score'] < threshold
        for hand in hands
    )


def _gap_plausible(before, after, steps, frame):
    """Conservative spatial checks adapted from the MediaPipe branch."""
    try:
        boxes = [np.asarray(h['bbox_px'], dtype=float) for h in (before, after)]
        width, height = float(frame['frame_width']), float(frame['frame_height'])
        if width <= 0 or height <= 0 or any(b.shape != (4,) or not np.isfinite(b).all() for b in boxes):
            return False
        sizes = [max(b[2] - b[0], b[3] - b[1]) for b in boxes]
        if min(sizes) <= 0 or not 0.5 <= sizes[1] / sizes[0] <= 2.0:
            return False
        for b in boxes:
            if b[0] <= width * .02 or b[1] <= height * .02 or b[2] >= width * .98 or b[3] >= height * .98:
                return False
        centres = [(b[:2] + b[2:]) / 2 for b in boxes]
        return np.linalg.norm(centres[1] - centres[0]) <= steps * sum(sizes) / 2
    except (KeyError, TypeError, ValueError):
        return False


def fill_gaps(records, limit=3):
    """Fill short plausible gaps; preserve measured fields and never mutate input."""
    result = copy.deepcopy(records)
    for side in ('left', 'right'):
        for field in ('kp2d_px', 'kp3d_hand_m'):
            previous = None
            for i, frame in enumerate(records):
                hand = next((h for h in frame['hands'] if h['side'] == side and h.get(field)), None)
                if hand is None:
                    continue
                if previous is not None:
                    j, before = previous
                    if 0 < i-j-1 <= limit and _gap_plausible(before, hand, i-j, records[j]):
                        times = [records[k]['t_ms'] for k in range(j, i+1)]
                        if any(b <= a for a, b in zip(times, times[1:])):
                            previous = (i, hand)
                            continue
                        start, end = np.asarray(before[field], dtype=float), np.asarray(hand[field], dtype=float)
                        if start.shape != end.shape or not np.isfinite(start).all() or not np.isfinite(end).all():
                            previous = (i, hand)
                            continue
                        pending = []
                        for k in range(j+1, i):
                            alpha = (records[k]['t_ms']-times[0])/(times[-1]-times[0])
                            points = (1-alpha)*start + alpha*end
                            target = next((h for h in result[k]['hands'] if h['side'] == side), None)
                            if target is None and field != 'kp2d_px':
                                continue
                            if target is None:
                                box = [*points.min(axis=0), *points.max(axis=0)]
                                if any(iou(box, h['bbox_px']) > .3 for h in result[k]['hands']):
                                    pending = []
                                    break
                            pending.append((k, target, points))
                        for k, target, points in pending:
                            if target is None:
                                target = copy.deepcopy(before)
                                target.update(kp3d_hand_m=None, handedness_score=None, source='interpolated')
                                target.pop('kp3d_hand_m_source', None)
                                result[k]['hands'].append(target)
                            target[field] = points.tolist()
                            target[field+'_source'] = 'interpolated'
                            if field == 'kp2d_px':
                                target['bbox_px'] = [*points.min(axis=0), *points.max(axis=0)]
                previous = (i, hand)
    return result


def run_ps1(video, output_dir):
    with INFERENCE_LOCK:
        return _run(video, Path(output_dir))


def _run(video, out):
    out.mkdir(parents=True, exist_ok=True)
    if not MODEL.is_file():
        raise RuntimeError('Hand model missing: run backend/setup_models.py')
    records = []
    engine = None
    fallback_frames = 0
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Video has invalid frame rate')
    crop_options = mp.vision.HandLandmarkerOptions(base_options=mp.mp_python.BaseOptions(model_asset_path=str(MODEL)), running_mode=mp.vision.RunningMode.IMAGE, num_hands=1)
    with mp.make_landmarker(MODEL) as detector, mp.vision.HandLandmarker.create_from_options(crop_options) as crop_detector:
        for idx, frame in mp.read_frames(video, step=1):
            rec = mp.process_frame(detector, frame, idx, fps, out.parent.name)
            originals = rec['hands']
            if not needs_rtm_fallback(originals):
                # RTM's previous-frame proposals must not carry across skipped frames.
                if engine is not None:
                    engine.prev_boxes = []
                    engine.prev_hands = []
                records.append(rec)
                continue
            if engine is None:
                engine = RTMHandEngine()
                print(f"[inference] runtime={engine.runtime}", flush=True)
            fallback_frames += 1
            from .rtm_hand import to_schema_hands
            refined = to_schema_hands(engine.estimate(frame, originals))
            used = set()
            for hand in refined:
                candidates = [(iou(hand['bbox_px'], h['bbox_px']), n, h) for n,h in enumerate(originals) if n not in used]
                match = max(candidates, key=lambda x:x[0], default=None)
                hand['kp2d_px_source'] = 'rtmpose'
                if match and match[0] >= .3:
                    _, n, original = match
                    used.add(n)
                    hand.update(side=original['side'], hand_id=original['hand_id'], kp3d_hand_m=original['kp3d_hand_m'])
                    hand['kp3d_hand_m_source'] = 'mediapipe'
                else:
                    hand.update(side='unknown', hand_id=f'U{len(used)+refined.index(hand)+1}', kp3d_hand_m=None)
                    x1,y1,x2,y2 = hand['bbox_px']
                    pad = max(x2-x1,y2-y1)*.4
                    crop = frame[max(0,int(y1-pad)):min(frame.shape[0],int(y2+pad)),max(0,int(x1-pad)):min(frame.shape[1],int(x2+pad))]
                    if crop.size:
                        result = crop_detector.detect(mp.mp.Image(image_format=mp.mp.ImageFormat.SRGB,data=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)))
                        if result.hand_world_landmarks:
                            hand['kp3d_hand_m'] = [[p.x,p.y,p.z] for p in result.hand_world_landmarks[0]]
                            hand['kp3d_hand_m_source'] = 'mediapipe_crop'
                            side = result.handedness[0][0].category_name.lower()
                            if side in ('left','right') and not any(h['side']==side for h in originals):
                                hand.update(side=side,hand_id='L' if side=='left' else 'R')
            # Keep measured 3D-bearing hands if refinement fails to match them.
            refined.extend(h for n,h in enumerate(originals) if n not in used)
            rec['hands'] = refined
            records.append(rec)
    if not records:
        raise ValueError('No decodable video frames')
    print(f"[inference] RTMPose fallback frames={fallback_frames}/{len(records)} runtime={engine.runtime if engine else 'not needed'}", flush=True)
    (out/'hands.raw.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records))
    corrected = correct_hand_sides(records, window=60)
    filled = fill_gaps(corrected)
    path = out/'hands.jsonl'
    path.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in filled))
    pose = {'schema_version':'1.0','clip_id':out.parent.name,'coordinate_frame':'hand_centered','frames':[
        {'t_ms':r['t_ms'],'hands':[{'hand_id':h['hand_id'],'side':h['side'],'joints_3d_m':h['kp3d_hand_m'],'source':'interpolated' if h.get('kp3d_hand_m_source') == 'interpolated' else h['source']} for h in r['hands'] if h.get('kp3d_hand_m')]} for r in filled]}
    (out/'ps4.json').write_text(json.dumps(pose,allow_nan=False))
    return path
