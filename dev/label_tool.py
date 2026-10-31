"""Mark hand-object contacts by hand (ground truth for evaluation). Raw video only: no predictions are ever drawn.

Usage: python dev/label_tool.py <video_path>

Writes labels/<clip>_contacts.csv after EVERY change (and labels/<clip>_contacts.pos with the last position), so you can
quit any time and run the same command again to continue.

KEYS (click the video window first)
  Left / Right    1 frame back / forward       , and .   also 1 frame
  Up / Down       1 second forward / back      [ and ]   also 1 second
  PageUp / PageDn 5 seconds forward / back     - and =   also 5 seconds
  Space           play / pause                 g         go to a time (type 75 or 1:15, Enter)
  s               mark the START of a contact (first frame with contact)
  e               mark the END of a contact (last frame with contact)
  c               save the marked contact: you answer the questions on the screen
  u               unsure span: press once at its start, again at its end (excluded from scoring)
  z               undo the last saved contact or unsure span
  h               show / hide this key map          q or Esc   save and quit

QUESTIONS after c (Enter = accept the value shown in brackets, which is your previous answer)
  hand (worker's own view: l/r) -> state (p portable, f fixed) -> interaction (t touch, h hold) ->
  object name (for a hold: the TOOL in the hand; Tab completes earlier names) ->
  instance tag (the SAME physical tool always gets the SAME tag, e.g. hammer1) -> unsure? (y/n, default n)
  Esc cancels the question list.

One row per hand per contact. A contact flagged unsure is excluded from scoring like an unsure span.
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
KEYS = {"left": 2424832, "up": 2490368, "right": 2555904, "down": 2621440, "pgup": 2162688, "pgdn": 2228224}
COLUMNS = ["kind", "start_ms", "end_ms", "hand", "state", "interaction", "object", "instance", "unsure"]
WIN_W = 1100
STRIP_H = 44

HELP = [
    "KEYS   Left/Right 1 frame   Up/Down 1 s   PgUp/PgDn 5 s   (also , . [ ] - =)   Space play   g go to time",
    "s mark START   e mark END   c save contact (then answer questions)   u unsure span start/end   z undo   h hide help   q quit",
    "Questions: Enter = accept [default]  |  hand l/r  state p/f  interaction t/h  |  names: type, Tab completes  |  Esc cancels",
]


class LabelSession:
    """The label file and the interval being marked. No window code here, so it can be tested."""

    def __init__(self, clip: str, fps: float, n_frames: int, path: Path):
        self.clip, self.fps, self.n_frames, self.path = clip, fps, n_frames, Path(path)
        self.rows: list[dict] = []
        self.start = self.end = self.span_start = None
        self.position = 0
        if self.path.exists():
            with self.path.open(encoding="utf-8", newline="") as f:
                self.rows = [dict(r) for r in csv.DictReader(f)]
        pos = self.path.with_suffix(".pos")
        if pos.exists():
            try:
                self.position = min(max(int(json.loads(pos.read_text(encoding="utf-8"))["frame"]), 0), n_frames - 1)
            except (ValueError, KeyError):
                pass

    # -- time
    def t_ms(self, frame: int) -> int:
        return round(frame * 1000 / self.fps)

    def end_ms(self, frame: int) -> int:
        """The END mark is the last frame WITH contact, so the interval ends where the next frame begins."""
        return int(round(self.t_ms(frame) + 1000 / self.fps))

    # -- marking
    def mark_start(self, frame: int) -> None:
        self.start = frame

    def mark_end(self, frame: int) -> None:
        self.end = frame

    def ready(self) -> bool:
        return self.start is not None and self.end is not None and self.end >= self.start

    def add_contact(self, hand: str, state: str, interaction: str, obj: str, instance: str = "", unsure: bool = False) -> dict:
        if not self.ready():
            raise ValueError("mark a start (s) and an end (e) first; the end must not be before the start")
        for name, value, allowed in (("hand", hand, {"left", "right"}), ("state", state, {"portable", "fixed"}),
                                     ("interaction", interaction, {"touch", "hold"})):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {sorted(allowed)}, not {value!r}")
        if not obj.strip():
            raise ValueError("the object name is needed (for a hold: the tool in the hand)")
        row = {"kind": "contact", "start_ms": self.t_ms(self.start), "end_ms": self.end_ms(self.end), "hand": hand,
               "state": state, "interaction": interaction, "object": obj.strip().lower(),
               "instance": instance.strip().lower(), "unsure": int(unsure)}
        self._add(row)
        self.start = self.end = None
        return row

    def toggle_unsure_span(self, frame: int) -> dict | None:
        """First call opens a span at `frame`; the second closes it (and saves it). Returns the saved row or None."""
        if self.span_start is None:
            self.span_start = frame
            return None
        first, last = sorted((self.span_start, frame))
        self.span_start = None
        row = {"kind": "unsure_span", "start_ms": self.t_ms(first), "end_ms": self.end_ms(last), "hand": "", "state": "",
               "interaction": "", "object": "", "instance": "", "unsure": 1}
        self._add(row)
        return row

    def _add(self, row: dict) -> None:
        self.rows.append(row)
        self.rows.sort(key=lambda r: (int(r["start_ms"]), r["kind"], r["hand"]))
        self.last_added = row
        self.save()

    def undo(self) -> dict | None:
        """Remove the most recently ADDED row (not the latest in time)."""
        row = getattr(self, "last_added", None)
        if row is None or row not in self.rows:
            return None
        self.rows.remove(row)
        self.last_added = None
        self.save()
        return row

    # -- file
    def save(self, position: int | None = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, COLUMNS)
            writer.writeheader()
            writer.writerows({k: r.get(k, "") for k in COLUMNS} for r in self.rows)
        tmp.replace(self.path)       # a crash while saving never leaves a half-written file
        if position is not None:
            self.position = position
        self.path.with_suffix(".pos").write_text(json.dumps({"frame": self.position}), encoding="utf-8")

    # -- what the next question should offer
    def names(self) -> list[str]:
        return list(dict.fromkeys(r["object"] for r in reversed(self.rows) if r["kind"] == "contact" and r["object"]))

    def tags_for(self, obj: str) -> list[str]:
        return list(dict.fromkeys(r["instance"] for r in reversed(self.rows)
                                  if r["kind"] == "contact" and r["object"] == obj and r["instance"]))

    def counts(self) -> tuple[int, int]:
        return (sum(r["kind"] == "contact" for r in self.rows), sum(r["kind"] == "unsure_span" for r in self.rows))


ENTER, ESC, BACKSPACE, TAB = 13, 27, 8, 9


class Form:
    """The question list that follows `c`, driven one key at a time (testable without a window)."""

    FIELDS = [("hand", "hand (worker's own, l/r)", {"l": "left", "r": "right"}),
              ("state", "state (p portable / f fixed)", {"p": "portable", "f": "fixed"}),
              ("interaction", "interaction (t touch / h hold)", {"t": "touch", "h": "hold"}),
              ("object", "object name (hold: the TOOL in hand)", None),
              ("instance", "instance tag (same tool = same tag)", None),
              ("unsure", "unsure? (y/n)", {"y": True, "n": False})]

    def __init__(self, defaults: dict, names: list[str], tags_for):
        self.defaults, self.names, self.tags_for = dict(defaults), names, tags_for
        self.values: dict = {}
        self.i = 0
        self.buffer = ""
        self.done = self.cancelled = False

    @property
    def field(self):
        return self.FIELDS[self.i]

    def default(self):
        key = self.field[0]
        if key == "instance":     # the tag last used with this object name, if any
            tags = self.tags_for(self.values.get("object", ""))
            return tags[0] if tags else ""
        return self.defaults.get(key, "")

    def prompt(self) -> str:
        key, label, choices = self.field
        shown = self.default()
        if key == "unsure":
            shown = "y" if shown is True else "n"
        hint = ""
        if key == "object":
            hint = "   earlier: " + ", ".join(self.names[:6])
        if key == "instance":
            hint = "   used for this object: " + ", ".join(self.tags_for(self.values.get("object", ""))[:5])
        return f"{label} [{shown}]: {self.buffer}{hint}"

    def key(self, k: int) -> None:
        name, _, choices = self.field
        if k == ESC:
            self.cancelled = True
            return
        if choices is not None:       # one-letter answer
            if k == ENTER:
                value = self.default() if self.default() != "" else (False if name == "unsure" else "")
                if value == "":
                    return           # nothing to accept yet
                self._accept(value)
            elif 0 < k < 128 and chr(k).lower() in choices:
                self._accept(choices[chr(k).lower()])
            return
        if k == ENTER:                # text answer
            value = self.buffer.strip().lower() or self.default()
            if name == "object" and not value:
                return               # an object name is needed
            self._accept(value)
        elif k == BACKSPACE:
            self.buffer = self.buffer[:-1]
        elif k == TAB:
            match = next((n for n in self.names if n.startswith(self.buffer.lower()) and n != self.buffer.lower()), None)
            if match:
                self.buffer = match
        elif 32 <= k < 127:
            self.buffer += chr(k)

    def _accept(self, value) -> None:
        self.values[self.field[0]] = value
        self.buffer = ""
        self.i += 1
        if self.i == len(self.FIELDS):
            self.done = True


class TimeEntry:
    """Typed time for `g`: seconds (75, 75.5) or m:ss (1:15)."""

    @staticmethod
    def parse(text: str) -> float | None:
        try:
            minutes, _, seconds = text.strip().rpartition(":")
            return int(minutes or 0) * 60 + float(seconds)
        except ValueError:
            return None


class Frames:
    """Random access to video frames with a small cache (stepping forward is cheap, seeking is not)."""

    def __init__(self, video: Path):
        self.cap = cv2.VideoCapture(str(video))
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        w, h = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.size = (WIN_W, int(h * WIN_W / w))
        self.pos = 0       # index of the next frame cap.read() will return
        self.cache: dict[int, np.ndarray] = {}

    def get(self, i: int) -> np.ndarray:
        i = min(max(i, 0), self.n - 1)
        if i not in self.cache:
            if i != self.pos:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, bgr = self.cap.read()
            self.pos = i + 1
            if not ok:
                return np.zeros((self.size[1], self.size[0], 3), np.uint8)
            self.cache[i] = cv2.resize(bgr, self.size)
            if len(self.cache) > 90:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[i]


def text(img, s, org, color=(255, 255, 255), scale=0.5, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def clock(ms: float) -> str:
    m, s = divmod(ms / 1000, 60)
    return f"{int(m):02d}:{s:06.3f}"


def draw(frames: Frames, s: LabelSession, idx: int, playing: bool, show_help: bool, prompt: str, notice: str) -> np.ndarray:
    img = frames.get(idx).copy()
    w, h = frames.size
    t = s.t_ms(idx)
    n_contacts, n_spans = s.counts()
    text(img, f"{clock(t)} / {clock(s.t_ms(frames.n - 1))}   frame {idx}/{frames.n - 1}   {'PLAYING' if playing else 'paused'}",
         (8, 22), (0, 255, 255), 0.6, 2)
    marks = f"start: {'-' if s.start is None else clock(s.t_ms(s.start))}   end: {'-' if s.end is None else clock(s.t_ms(s.end))}" \
            f"   unsure span: {'open since ' + clock(s.t_ms(s.span_start)) if s.span_start is not None else '-'}" \
            f"   saved: {n_contacts} contacts, {n_spans} unsure spans"
    text(img, marks, (8, 46), (255, 255, 255), 0.5)
    y = 70
    for r in s.rows:        # what is already labelled at this moment
        if int(r["start_ms"]) <= t < int(r["end_ms"]):
            label = "UNSURE SPAN" if r["kind"] == "unsure_span" else \
                f"{r['hand']} {r['state']} {r['interaction']} {r['object']} {r['instance']}{' (unsure)' if str(r['unsure']) == '1' else ''}"
            text(img, "labelled here: " + label, (8, y), (0, 200, 255), 0.5)
            y += 20
    if notice:
        text(img, notice, (8, h - 78 if show_help else h - 40), (0, 255, 0), 0.55, 1)
    if prompt:
        cv2.rectangle(img, (0, h - 34), (w, h), (0, 0, 0), -1)
        text(img, prompt, (8, h - 12), (0, 255, 255), 0.6, 2)
    elif show_help:
        cv2.rectangle(img, (0, h - 62), (w, h), (0, 0, 0), -1)
        for k, line in enumerate(HELP):
            text(img, line, (6, h - 44 + 19 * k), (220, 220, 220), 0.45)
    strip = np.full((STRIP_H, w, 3), 30, np.uint8)
    total = frames.n * 1000 / frames.fps
    for r in s.rows:
        x1, x2 = int(int(r["start_ms"]) / total * w), int(int(r["end_ms"]) / total * w)
        if r["kind"] == "unsure_span":
            cv2.rectangle(strip, (x1, 2), (max(x2, x1 + 1), 8), (0, 0, 200), -1)
        else:
            row_y = 12 if r["hand"] == "left" else 28
            color = (0, 200, 0) if r["interaction"] == "hold" else (255, 150, 0)
            cv2.rectangle(strip, (x1, row_y), (max(x2, x1 + 1), row_y + 12), color, -1)
    cx = int(t / total * w)
    cv2.line(strip, (cx, 0), (cx, STRIP_H), (255, 255, 255), 2)
    return np.vstack([img, strip])


def main(video_path: str) -> None:
    video = Path(video_path)
    frames = Frames(video)
    s = LabelSession(video.stem, frames.fps, frames.n, ROOT / "labels" / f"{video.stem}_contacts.csv")
    idx, playing, show_help, notice = s.position, False, True, ""
    last = {"hand": "right", "state": "portable", "interaction": "hold", "object": "", "unsure": False}
    form = entry = None          # an open question list, or the typed time of `g`
    fps = frames.fps
    jump = {KEYS["right"]: 1, ord("."): 1, KEYS["left"]: -1, ord(","): -1,
            KEYS["up"]: round(fps), ord("]"): round(fps), KEYS["down"]: -round(fps), ord("["): -round(fps),
            KEYS["pgup"]: round(5 * fps), ord("="): round(5 * fps), KEYS["pgdn"]: -round(5 * fps), ord("-"): -round(5 * fps)}
    cv2.namedWindow("label", cv2.WINDOW_AUTOSIZE)
    while True:
        prompt = form.prompt() if form else ("go to (seconds or m:ss): " + entry) if entry is not None else ""
        cv2.imshow("label", draw(frames, s, idx, playing, show_help, prompt, notice))
        key = cv2.waitKeyEx(int(1000 / fps) if playing and not prompt else 0)
        if cv2.getWindowProperty("label", cv2.WND_PROP_VISIBLE) < 1:
            break
        if playing and key == -1:
            idx = min(idx + 1, frames.n - 1)
            continue
        if form:
            form.key(key)
            if form.cancelled:
                form, notice = None, "cancelled"
            elif form.done:
                v = form.values
                try:
                    row = s.add_contact(v["hand"], v["state"], v["interaction"], v["object"], v["instance"], bool(v["unsure"]))
                    last = {"hand": v["hand"], "state": v["state"], "interaction": v["interaction"], "object": v["object"],
                            "unsure": False}
                    notice = f"saved: {row['hand']} {row['state']} {row['interaction']} {row['object']} {row['instance']}"
                except ValueError as e:
                    notice = str(e)
                form = None
            continue
        if entry is not None:
            if key == ENTER:
                seconds = TimeEntry.parse(entry)
                if seconds is not None:
                    idx = min(max(round(seconds * fps), 0), frames.n - 1)
                entry = None
            elif key == ESC:
                entry = None
            elif key == BACKSPACE:
                entry = entry[:-1]
            elif 32 <= key < 127:
                entry += chr(key)
            continue
        ch = chr(key) if 0 <= key < 128 else ""
        notice = ""
        if key in jump:
            idx = min(max(idx + jump[key], 0), frames.n - 1)
        elif ch == " ":
            playing = not playing
        elif ch == "s":
            s.mark_start(idx)
            notice = f"start marked at {clock(s.t_ms(idx))}"
        elif ch == "e":
            s.mark_end(idx)
            notice = f"end marked at {clock(s.t_ms(idx))}"
        elif ch == "c":
            if not s.ready():
                notice = "mark a start (s) and an end (e) first; the end must not be before the start"
            else:
                form = Form(last, s.names(), s.tags_for)
        elif ch == "u":
            row = s.toggle_unsure_span(idx)
            notice = "unsure span saved" if row else f"unsure span opened at {clock(s.t_ms(idx))}; press u again at its end"
        elif ch == "z":
            row = s.undo()
            notice = f"removed {row['kind']} {clock(int(row['start_ms']))}" if row else "nothing to undo"
        elif ch == "g":
            entry = ""
        elif ch == "h":
            show_help = not show_help
        elif ch == "q" or key == ESC:
            break
        s.position = idx
    s.save(idx)
    cv2.destroyAllWindows()
    n_contacts, n_spans = s.counts()
    print(f"{n_contacts} contacts and {n_spans} unsure spans saved in {s.path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
