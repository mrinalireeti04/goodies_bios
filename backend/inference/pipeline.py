"""Local hybrid inference. Raw measurements are retained; only bounded gaps are filled."""
import copy
import json
import os
import subprocess
import threading
from pathlib import Path

import cv2
import numpy as np
from . import hand_detect as mp
from .labels import correct_hand_sides
from .rtm_hand import RTMHandEngine, iou

INFERENCE_LOCK = threading.Lock()
MODEL = Path(__file__).parent / 'models' / 'hand_landmarker.task'


def fill_gaps(records, limit=10):
    result = copy.deepcopy(records)
    # Anatomical side is the conservative identity here; never interpolate unknowns.
    for side in ('left', 'right'):
        for field in ('kp2d_px', 'kp3d_hand_m'):
            previous = None
            for i, frame in enumerate(records):
                hand = next((h for h in frame['hands'] if h['side'] == side and h.get(field)), None)
                if hand is None:
                    continue
                if previous is not None:
                    j, before = previous
                    if 0 < i-j-1 <= limit:
                        for k in range(j+1, i):
                            alpha = (records[k]['t_ms']-records[j]['t_ms'])/(frame['t_ms']-records[j]['t_ms'])
                            target = next((h for h in result[k]['hands'] if h['side'] == side), None)
                            if target is None:
                                if field != 'kp2d_px':
                                    continue
                                target = copy.deepcopy(before)
                                target['kp3d_hand_m'] = None
                                result[k]['hands'].append(target)
                            target[field] = ((1-alpha)*np.array(before[field])+alpha*np.array(hand[field])).tolist()
                            target['source'] = 'interpolated'
                            target[field+'_source'] = 'interpolated'
                            if field == 'kp2d_px':
                                pts = np.array(target[field])
                                target['bbox_px'] = [*pts.min(axis=0), *pts.max(axis=0)]
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
    engine = RTMHandEngine()
    print(f"[inference] runtime={engine.runtime}", flush=True)
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
    print(f"[inference] completed RTMPose runtime={engine.runtime}", flush=True)
    (out/'hands.raw.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records))
    corrected = correct_hand_sides(records, window=60)
    filled = fill_gaps(corrected)
    path = out/'hands.jsonl'
    path.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in filled))
    pose = {'schema_version':'1.0','clip_id':out.parent.name,'coordinate_frame':'hand_centered','frames':[
        {'t_ms':r['t_ms'],'hands':[{'hand_id':h['hand_id'],'side':h['side'],'joints_3d_m':h['kp3d_hand_m'],'source':h['source']} for h in r['hands'] if h.get('kp3d_hand_m')]} for r in filled]}
    (out/'ps4.json').write_text(json.dumps(pose,allow_nan=False))
    w,h = records[0]['frame_width'],records[0]['frame_height']
    cmd = ['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}','-r',str(fps),'-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-movflags','+faststart',str(out/'overlay.mp4')]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for idx, frame in mp.read_frames(video, step=1):
            proc.stdin.write(mp.draw_overlay(frame, filled[idx]['hands']).tobytes())
    finally:
        proc.stdin.close()
        code = proc.wait()
    if code:
        raise RuntimeError('Overlay encoding failed')
    return path
