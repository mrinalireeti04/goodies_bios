"""Check PS2's predictions by eye (dev only): each predicted contact, or each safety event with --safety, is looped from
0.5 s before to 0.5 s after (capped at 10 s) with the hand skeleton and the predicted object's box drawn.

Usage: python dev/verify_contacts.py <video_path> <ps2.json> [--safety] [--hands PATH]
       python dev/verify_contacts.py <video_path> --checklist [PATH]    (walk outputs/<clip>/checklist.json)

With --checklist you walk the check list built by dev/checklist.py. Each item shows only its SOURCE (system / gemini_pro /
both) and that source's own guess, never what the other models said, so your answer is not biased. For an item our system
missed (source gemini_pro) "y" means it was a real contact.

Keys (click the video window first):
  y  correct                                  n  no real contact (safety: false alarm)
  o  contact is real but the object is wrong (you then type the right name in the terminal)
  i  contact and object are right but touch/hold is wrong
  s  skip (shown again next time)             q  save and quit
Answers are saved at once to labels/verify_<clip>.csv (gitignored) and the run is resumable: items already answered are
not shown again. At the end it prints precision, naming accuracy and interaction accuracy as counts.
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

COLUMNS = ["kind", "key", "id", "start_ms", "end_ms", "hand_id", "object_id", "object_label", "interaction",
           "verdict", "correct_name", "source", "tier"]
VERDICTS = {"contact": {"y", "n", "o", "i"}, "safety": {"y", "n"}, "check": {"y", "n", "o", "i"}}
MARGIN_MS, MAX_LOOP_MS = 500, 10_000
WIN_W = 1100


def item_key(kind: str, item: dict) -> str:
    """Stable across re-runs that give the same interval (event ids are renumbered by every run)."""
    if kind == "check":
        return f"check|{item['key']}"
    hand = item["hand_id"]
    return f"{kind}|{hand}|{item['start_ms']}|{item['end_ms']}"


class VerifySession:
    """The answers file and the statistics. No window code here, so it can be tested."""

    def __init__(self, kind: str, items: list[dict], csv_path: Path):
        self.kind, self.items, self.path = kind, items, Path(csv_path)
        self.rows: list[dict] = []     # answers of this kind
        self.others: list[dict] = []   # answers of the other kind in the same file, kept untouched
        if self.path.exists():
            with self.path.open(encoding="utf-8", newline="") as f:
                for r in csv.DictReader(f):
                    (self.rows if r["kind"] == kind else self.others).append(r)
        self.answered = {r["key"] for r in self.rows}

    def pending(self) -> list[dict]:
        return [it for it in self.items if item_key(self.kind, it) not in self.answered]

    def record(self, item: dict, verdict: str, correct_name: str = "") -> dict:
        if verdict not in VERDICTS[self.kind]:
            raise ValueError(f"verdict {verdict!r} is not valid for {self.kind}")
        if verdict == "o" and not correct_name.strip():
            raise ValueError("verdict 'o' needs the right object name")
        row = {"kind": self.kind, "key": item_key(self.kind, item), "id": item.get("event_id", ""),
               "start_ms": item["start_ms"], "end_ms": item["end_ms"], "hand_id": item["hand_id"],
               "object_id": item.get("object_id") or item.get("hazard_object_id") or "",
               "object_label": item.get("object_label", ""), "interaction": item.get("interaction", ""),
               "verdict": verdict, "correct_name": correct_name.strip(), "source": item.get("source", ""), "tier": item.get("tier", "")}
        self.rows = [r for r in self.rows if r["key"] != row["key"]] + [row]
        self.answered.add(row["key"])
        self.save()
        return row

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, COLUMNS)
            writer.writeheader()
            for r in self.others + self.rows:
                writer.writerow({k: r.get(k, "") for k in COLUMNS})

    def stats(self) -> dict[str, tuple[int, int]]:
        """(hits, total) pairs. Contacts: precision = real / judged; naming = right object / real;
        interaction = right touch/hold / real. Safety: precision = real / judged."""
        v = [r["verdict"] for r in self.rows]
        y, n, o, i = (v.count(x) for x in "yno" + "i")
        if self.kind == "safety":
            return {"precision": (y, y + n)}
        judged, real = y + n + o + i, y + o + i
        return {"precision": (real, judged), "naming": (y + i, real), "interaction": (y + o, real)}


def format_stats(stats: dict[str, tuple[int, int]]) -> str:
    return "\n".join(f"{name}: {h}/{t}" + (f" ({100 * h / t:.0f}%)" if t else "") for name, (h, t) in stats.items())


def load_segment(cap, fps: float, item: dict, n_frames: int) -> tuple[int, list[np.ndarray]]:
    """Frames from 0.5 s before to 0.5 s after the item (at most MAX_LOOP_MS), shrunk for the window."""
    start = max(item["start_ms"] - MARGIN_MS, 0)
    end = min(item["end_ms"] + MARGIN_MS, item["start_ms"] + MAX_LOOP_MS)
    i0, i1 = int(start * fps / 1000), min(int(end * fps / 1000), n_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
    frames = []
    for _ in range(i0, i1):
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr)
    return i0, frames


SOURCE_NAMES = {"system": "system", "gemini_pro": "Gemini Pro", "both": "system + Gemini Pro"}


def checklist_items(checklist: dict) -> list[dict]:
    """The check list as the window shows it: the item's source and that source's own guess. Votes of the other models
    (and Groq) are deliberately left out so they cannot bias the answer."""
    out = []
    for it in checklist["items"]:
        system, gemini = it["system"], it["gemini"]
        out.append({"event_id": it["id"], "key": it["key"], "start_ms": it["start_ms"], "end_ms": it["end_ms"],
                    "hand_id": it["hand"], "object_id": (system or {}).get("object_id") or "",
                    "object_label": system["object_label"] if system else gemini["object"],
                    "interaction": system["interaction"] if system else "", "source": it["source"], "tier": it["tier"]})
    return out


def main(video_path: str, result_path: str | None, safety: bool = False, hands_path: str | None = None,
         checklist_path: str | None = None) -> None:
    from render import HAND_COLORS, draw_skeleton, text
    from ps2.hands import load_hands

    video = Path(video_path)
    if checklist_path:
        checklist = json.loads(Path(checklist_path).read_text(encoding="utf-8"))
        result = json.loads((ROOT / "outputs" / video.stem / f"{checklist['variant']}.json").read_text(encoding="utf-8"))
        kind, items = "check", checklist_items(checklist)
    else:
        result = json.loads(Path(result_path).read_text(encoding="utf-8"))
        kind = "safety" if safety else "contact"
        items = result.get("safety_events", []) if safety else result["contacts"]
    session = VerifySession(kind, items, ROOT / "labels" / f"verify_{result['clip_id']}.csv")
    todo = session.pending()
    print(f"{len(items)} {kind} items, {len(items) - len(todo)} already answered, {len(todo)} to go. Keys: y n o i s q")
    if not todo:
        print(format_stats(session.stats()))
        return

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames, vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    hands = load_hands(Path(hands_path) if hands_path else ROOT / "outputs" / video.stem / "ps1_stub.jsonl", vw, vh)
    frames_meta = result["frames"]
    scale = WIN_W / vw
    hand_ids = sorted({hs["hand_id"] for f in frames_meta for hs in f["hand_states"]})
    colors = {h: HAND_COLORS[i % len(HAND_COLORS)] for i, h in enumerate(hand_ids)}
    quit_now = False
    for n, item in enumerate(todo, 1):
        i0, frames = load_segment(cap, fps, item, n_frames)
        if not frames:
            continue
        oid = item.get("object_id") or item.get("hazard_object_id")
        desc = (f"{item['hand_id']} {item.get('interaction', '')} {item.get('object_label', '')}" if not safety
                else f"safety: {item['hand_id']} near {oid} (hazard moving)")
        if kind == "check":
            desc = f"source: {SOURCE_NAMES[item['source']]} | {item['hand_id']} hand | {item['object_label']}"
        verdict = None
        while verdict is None:
            for k, bgr in enumerate(frames):
                idx = i0 + k
                img = cv2.resize(bgr, (WIN_W, int(vh * scale)))
                for o in (frames_meta[idx]["objects"] if idx < len(frames_meta) else []):
                    if o["object_id"] == oid:
                        x1, y1, x2, y2 = (int(v * scale) for v in o["bbox_px"])
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255) if safety else (0, 255, 255), 3)
                for hand in hands.get(idx, []):
                    draw_skeleton(img, hand.kp, scale, colors.get(hand.hand_id, (255, 255, 255)))
                inside = item["start_ms"] <= frames_meta[min(idx, len(frames_meta) - 1)]["t_ms"] < item["end_ms"]
                text(img, f"[{n}/{len(todo)}] {desc}  {('<< CANDIDATE' if kind == 'check' else '<< PREDICTED') if inside else ''}", (8, 24), (0, 255, 255), 0.6, 2)
                text(img, "y correct  n no  o wrong object  i wrong touch/hold  s skip  q quit" if not safety
                     else "y real  n false alarm  s skip  q quit", (8, img.shape[0] - 10), (255, 255, 255), 0.5)
                cv2.imshow("verify", img)
                key = cv2.waitKey(int(1000 / fps)) & 0xFF
                ch = chr(key) if 0 < key < 128 else ""
                if ch in VERDICTS[kind] | {"s", "q"}:
                    verdict = ch
                    break
            if cv2.getWindowProperty("verify", cv2.WND_PROP_VISIBLE) < 1:
                verdict = "q"
        if verdict == "q":
            quit_now = True
            break
        if verdict == "s":
            continue
        name = ""
        if verdict == "o":
            while not name:
                name = input(f"right object name for {item.get('object_label', '')}: ").strip()
        session.record(item, verdict, name)
    cv2.destroyAllWindows()
    print(format_stats(session.stats()))
    if quit_now:
        print("stopped early; run again to continue")


if __name__ == "__main__":
    argv = sys.argv[1:]
    hands_arg = argv[argv.index("--hands") + 1] if "--hands" in argv else None
    checklist = None
    if "--checklist" in argv:
        k = argv.index("--checklist")
        checklist = argv[k + 1] if k + 1 < len(argv) and not argv[k + 1].startswith("--") else "auto"
    skip = {"--hands", "--checklist"}
    positional = [a for k, a in enumerate(argv) if not a.startswith("--") and not (k > 0 and argv[k - 1] in skip)]
    if checklist == "auto" and positional:
        checklist = str(ROOT / "outputs" / Path(positional[0]).stem / "checklist.json")
    if len(positional) != (1 if checklist else 2):
        sys.exit(__doc__)
    main(positional[0], None if checklist else positional[1], "--safety" in argv, hands_arg, checklist)
