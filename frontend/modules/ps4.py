"""PS4 minimum: turn PS1 hand-centered world landmarks into 3D pose data and an MP4.

No image/video footage is read here. The 3D points are *relative to the hand center*,
not a world-space factory trajectory. Keep outputs private per competition rules.

Usage:
    python -m modules.ps4 path/to/hands.jsonl path/to/private_output
Or:
    from modules.ps4 import run_ps4
    result_path = run_ps4(Path('hands.jsonl'), Path('private_output'))
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


def _read_ps1(path: Path) -> list[dict]:
    frames = []
    with path.open('r', encoding='utf-8') as file:
        for line_no, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid PS1 JSONL on line {line_no}') from exc
            if not isinstance(row.get('hands'), list) or not isinstance(row.get('t_ms'), int):
                raise ValueError(f'Line {line_no}: missing hands[] or integer t_ms')
            frames.append(row)
    if not frames:
        raise ValueError('PS1 input has no frames')
    if len({f.get('clip_id') for f in frames}) != 1 or not frames[0].get('clip_id'):
        raise ValueError('All PS1 frames must share the same clip_id')
    frames.sort(key=lambda f: f['t_ms'])
    if any(a['t_ms'] >= b['t_ms'] for a,b in zip(frames,frames[1:])):
        raise ValueError('PS1 frame timestamps must be strictly increasing')
    return frames


def _to_ps4(frames: list[dict]) -> list[dict]:
    converted = []
    for frame in frames:
        hands = []
        for hand in frame['hands']:
            joints = hand.get('kp3d_hand_m')
            if joints is None:
                continue
            if len(joints)!=21 or any(len(p)!=3 or not all(isinstance(v,(int,float)) and abs(v)<1e3 for v in p) for p in joints):
                raise ValueError(f"Invalid 3D landmarks at {frame['t_ms']} ms")
            hands.append({
                'hand_id': str(hand['hand_id']),
                'side': hand.get('side','unknown'),
                'joints_3d_m': joints,
                'source': hand.get('source','detected'),
            })
        converted.append({'frame_idx':frame.get('frame_idx',len(converted)), 't_ms':frame['t_ms'], 'hands':hands})
    return converted


def _fps(frames: list[dict]) -> float:
    gaps = [b['t_ms']-a['t_ms'] for a,b in zip(frames,frames[1:])]
    return min(60., max(1., round(1000./statistics.median(gaps),3))) if gaps else 25.


def _render_video(frames: list[dict], destination: Path, fps: float) -> bool:
    """Lightweight 3D perspective projection with X/Y/Z axes; no source frames used."""
    try:
        import cv2
        import numpy as np
    except Exception:
        # Optional MP4 rendering must never block the structured PS4 JSON output.
        return False
    width,height=960,460
    writer=cv2.VideoWriter(str(destination),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
    if not writer.isOpened():
        return False
    try:
        for frame in frames:
            img=np.full((height,width,3),(26,34,48),dtype=np.uint8)
            for panel,side in enumerate(('left','right')):
                cx=width//4+panel*width//2
                cv2.putText(img, f'{side.upper()} HAND', (cx-190,46),cv2.FONT_HERSHEY_SIMPLEX,.72,(210,225,240),2)
                cv2.putText(img, 'Hand-centered 3D · axes not world-anchored', (cx-195,height-23),cv2.FONT_HERSHEY_SIMPLEX,.43,(140,163,185),1)
                hands=[h for h in frame['hands'] if h['side']==side]
                if not hands:
                    cv2.putText(img,'No valid 3D pose',(cx-145,235),cv2.FONT_HERSHEY_SIMPLEX,.67,(159,167,180),2)
                    continue
                hand=hands[0]
                xyz=np.array(hand['joints_3d_m'],dtype=float)
                # Keep a consistent projection scale: 1 meter = 2100 screen pixels.
                # This is a display-only isometric view, NOT an estimated factory pose.
                P=np.array([[1.,0.,-.46],[0.,-1.,-.24]])
                xy=xyz@P.T * 2100
                pts=[(int(cx+x),int(292+y)) for x,y in xy]
                color=(98,216,153) if side=='right' else (75,166,244)
                for a,b in CONNECTIONS: cv2.line(img,pts[a],pts[b],color,2,cv2.LINE_AA)
                for i,p in enumerate(pts):cv2.circle(img,p,4,(230,245,250) if i==0 else color,-1,cv2.LINE_AA)
                cv2.putText(img,f"Track {hand['hand_id']}",(cx-190,80),cv2.FONT_HERSHEY_SIMPLEX,.52,(174,193,208),1)
            cv2.putText(img,f"t = {frame['t_ms']/1000:.3f} s", (width//2-84,31),cv2.FONT_HERSHEY_SIMPLEX,.63,(230,239,244),2)
            writer.write(img)
    finally:
        writer.release()
    return destination.exists() and destination.stat().st_size>0


def run_ps4(hands_path: Path, output_dir: Path) -> Path:
    """Return ps4.json; optionally writes hands_3d.mp4 if OpenCV is available."""
    hands_path,output_dir=Path(hands_path),Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    frames=_read_ps1(hands_path)
    poses=_to_ps4(frames)
    fps=_fps(frames)
    video=output_dir/'hands_3d.mp4'
    rendered=_render_video(poses,video,fps)
    result={
        'schema_version':'1.0','clip_id':frames[0]['clip_id'],
        'coordinate_frame':'hand_centered',
        'source_file':hands_path.name,
        'render_video_path':str(video) if rendered else None,
        'render_fps':fps,
        'frames':poses,
    }
    dest=output_dir/'ps4.json'
    dest.write_text(json.dumps(result,indent=2),encoding='utf-8')
    return dest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Convert PS1 landmarks to 3D hand-centered animation')
    parser.add_argument('hands_path',type=Path)
    parser.add_argument('output_dir',type=Path)
    args=parser.parse_args()
    print(run_ps4(args.hands_path,args.output_dir))
