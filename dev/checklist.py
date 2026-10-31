"""Pool the system's contacts with the independent Gemini Pro proposals, let Groq judge every pooled item, and build the
check list a human verifies (dev only). No manual labelling of whole clips is needed: you only answer ~40 items.

Usage: python dev/checklist.py <video_path> [--variant ledger_sam] [--hands PATH]
  needs outputs/<clip>/<variant>.json (the system) and outputs/<clip>/independent.json (dev/independent_proposals.py);
  asks Groq about every pooled item (cached), then writes outputs/<clip>/checklist.json.

Items: a system contact and a proposal are "the same contact" if their time IoU >= checklist_iou and the hands are compatible.
Votes: system (a contact exists / not), Gemini Pro (a proposal exists / not), Groq (yes / no / unsure from a few frames).
Tier 1 = the three disagree completely (all different, in the vote or in the tool name); tier 2 = any other disagreement;
tier 3 = unanimous (everyone says contact, names agree). The list takes tier 1 first, then tier 2, then a random sample of
tier 3; a tier that does not fit is sampled at random (fixed seed), so the stratified estimates in evaluate.py stay valid.
"""
import base64
import json
import logging
import random
import re
import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

log = logging.getLogger("ps2")
AGREEMENTS_MIN = 8   # the list always keeps this many tier-3 items (when there are that many) to measure the "easy" cases


# ---------------------------------------------------------------- names and hands

def canonical(name: str) -> str:
    words = re.sub(r"[^a-z0-9 ]+", " ", str(name).lower().replace("_", " ")).split()
    if words and len(words[-1]) > 3 and words[-1].endswith("s") and not words[-1].endswith("ss"):
        words[-1] = words[-1][:-1]
    return " ".join(words)


def load_name_map(path: Path) -> list[set[str]]:
    """labels/name_map.yaml: {canonical: [synonym, ...]} -> groups of names that mean the same thing."""
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [{canonical(k), *(canonical(s) for s in v or [])} for k, v in data.items()]


def names_match(a: str, b: str, groups: list[set[str]] | None = None) -> bool:
    """Same tool name: equal, one contains the other ('claw hammer' ~ 'hammer'), or in one name_map group."""
    a, b = canonical(a), canonical(b)
    if not a or not b or "unknown" in (a, b):
        return False
    if a == b or a in b.split() or b in a.split() or a in b or b in a:
        return True
    return any(a in g and b in g for g in groups or [])


def hand_of(hand_id: str) -> str:
    h = str(hand_id).lower()
    return "left" if "left" in h else "right" if "right" in h else "unknown"


def hands_compatible(a: str, b: str) -> bool:
    return "unknown" in (a, b) or "both" in (a, b) or a == b


def interval_iou(a: dict, b: dict) -> float:
    inter = min(a["end_ms"], b["end_ms"]) - max(a["start_ms"], b["start_ms"])
    union = (a["end_ms"] - a["start_ms"]) + (b["end_ms"] - b["start_ms"]) - max(inter, 0)
    return max(inter, 0) / union if union > 0 else 0.0


# ---------------------------------------------------------------- pooling and tiers

def pool(system: list[dict], proposals: list[dict], iou_min: float) -> list[dict]:
    """One item per distinct contact. Greedy one-to-one matching, best overlap first."""
    pairs = sorted(((interval_iou(s, p), i, j) for i, s in enumerate(system) for j, p in enumerate(proposals)
                    if hands_compatible(hand_of(s["hand_id"]), p["hand"])), reverse=True)
    used_s, used_p, matched = set(), set(), {}
    for score, i, j in pairs:
        if score < iou_min:
            break
        if i not in used_s and j not in used_p:
            used_s.add(i)
            used_p.add(j)
            matched[i] = j
    items = []
    for i, s in enumerate(system):
        p = proposals[matched[i]] if i in matched else None
        items.append({"source": "both" if p else "system", "system": s, "gemini": p})
    items += [{"source": "gemini_pro", "system": None, "gemini": p} for j, p in enumerate(proposals) if j not in used_p]
    for it in items:
        primary = it["system"] or it["gemini"]
        it["start_ms"], it["end_ms"] = primary["start_ms"], primary["end_ms"]
        it["hand"] = hand_of(it["system"]["hand_id"]) if it["system"] and hand_of(it["system"]["hand_id"]) != "unknown" \
            else (it["gemini"]["hand"] if it["gemini"] else "unknown")
        it["key"] = f"{it['source']}|{it['start_ms']}|{it['end_ms']}"
    return sorted(items, key=lambda it: (it["start_ms"], it["end_ms"]))


def votes(item: dict, groq: dict | None) -> dict:
    """{'system','gemini','groq': yes|no|unsure|None, 'names': {judge: name}}"""
    names = {}
    if item["system"]:
        names["system"] = item["system"].get("object_label") or ""
    if item["gemini"]:
        names["gemini"] = item["gemini"]["object"]
    if groq and groq["contact"] == "yes":
        names["groq"] = groq["tool"]
    return {"system": "yes" if item["system"] else "no", "gemini": "yes" if item["gemini"] else "no",
            "groq": groq["contact"] if groq else None, "names": names}


def tier_of(v: dict, groups=None) -> int:
    """1 = complete disagreement, 2 = any other disagreement, 3 = unanimous."""
    cast = [v["system"], v["gemini"]] + ([v["groq"]] if v["groq"] else [])
    if len(set(cast)) == 3:
        return 1
    named = [n for n in v["names"].values() if n and canonical(n) != "unknown"]
    name_clash = any(not names_match(a, b, groups) for k, a in enumerate(named) for b in named[k + 1:])
    if len(set(cast)) == 1 and cast[0] == "yes" and not name_clash:
        return 3
    if len(set(cast)) == 1 and cast[0] == "yes":
        pairwise_distinct = len(named) == 3 and all(not names_match(a, b, groups) for k, a in enumerate(named) for b in named[k + 1:])
        return 1 if pairwise_distinct else 2
    return 2


def select(items: list[dict], tiers: list[int], target: int, seed: int) -> tuple[list[dict], dict]:
    """The check list: tier 1, then tier 2, then tier-3 agreements; a tier that does not fit is sampled at random.
    Returns (chosen items in tier order, {tier: {N, n}})."""
    rng = random.Random(seed)
    by_tier = {t: [it for it, tt in zip(items, tiers) if tt == t] for t in (1, 2, 3)}
    reserve = min(AGREEMENTS_MIN, len(by_tier[3]))
    room = max(target - reserve, 0)
    take = {}
    for t in (1, 2):
        n = min(len(by_tier[t]), room)
        take[t] = n
        room -= n
    take[3] = min(len(by_tier[3]), target - take[1] - take[2])
    chosen, strata = [], {}
    for t in (1, 2, 3):
        pick = by_tier[t] if take[t] >= len(by_tier[t]) else rng.sample(by_tier[t], take[t])
        chosen += sorted(pick, key=lambda it: it["start_ms"])
        strata[str(t)] = {"N": len(by_tier[t]), "n": len(pick)}
    return chosen, strata


def build_checklist(clip: str, variant: str, system: list[dict], proposals: list[dict], groq_answers: dict, cfg: dict,
                    groups=None, meta: dict | None = None, status_of: dict | None = None) -> dict:
    """status_of: object_id -> {"identified": bool, "flagged": bool} from the system output (tools[] / review_queue[])."""
    items = pool(system, proposals, cfg["checklist_iou"])
    all_votes = [votes(it, groq_answers.get(it["key"])) for it in items]
    tiers = [tier_of(v, groups) for v in all_votes]
    chosen_items, strata = select(items, tiers, cfg["checklist_size"], cfg["checklist_seed"])
    index = {id(it): k for k, it in enumerate(items)}
    out_items = []
    for n, it in enumerate(chosen_items, 1):
        k = index[id(it)]
        out_items.append({"id": f"k{n}", "key": it["key"], "tier": tiers[k], "source": it["source"], "start_ms": it["start_ms"],
                          "end_ms": it["end_ms"], "hand": it["hand"], "system": it["system"], "gemini": it["gemini"],
                          "groq": groq_answers.get(it["key"]), "votes": all_votes[k]})
    status_of = status_of or {}
    population = [{"key": it["key"], "tier": tiers[k], "source": it["source"], "votes": all_votes[k],
                   "identified": bool(it["system"] and status_of.get(it["system"]["object_id"], {}).get("identified")),
                   "flagged": bool(it["system"] and status_of.get(it["system"]["object_id"], {}).get("flagged"))}
                  for k, it in enumerate(items)]
    return {"clip": clip, "variant": variant, "seed": cfg["checklist_seed"], "target": cfg["checklist_size"],
            "iou": cfg["checklist_iou"], "strata": strata, "n_pooled": len(items), "items": out_items,
            "population": population, **(meta or {})}


# ---------------------------------------------------------------- frames for Groq

def frames_for_item(cap: cv2.VideoCapture, item: dict, n: int, fps: float, max_px: int = 768) -> list[str]:
    """n base64 JPEGs (start, middle, end) of the item's span; shrunk to at most max_px."""
    span = item["end_ms"] - item["start_ms"]
    moments = [item["start_ms"] + span * f for f in ((0.0, 0.5, 0.97) if n == 3 else (0.5,))]
    out = []
    for ms in moments:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(ms * fps / 1000), 0))
        ok, bgr = cap.read()
        if not ok:
            continue
        s = max_px / max(bgr.shape[:2])
        if s < 1:
            bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            out.append(base64.b64encode(jpg.tobytes()).decode())
    return out


def main(video_path: str, variant: str = "ledger_sam") -> None:
    from groq_judge import GroqJudge, judge_items
    from ps2.pipeline import load_config

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    video = Path(video_path)
    cfg = load_config()
    out_dir = ROOT / "outputs" / video.stem
    result = json.loads((out_dir / f"{variant}.json").read_text(encoding="utf-8"))
    system = result["contacts"]
    reviewed = {r["object_id"] for r in result.get("review_queue", []) if r["object_id"]}
    status_of = {t["object_id"]: {"identified": t["label_status"] == "identified", "flagged": t["label_status"] == "unidentified" or t["object_id"] in reviewed}
                 for t in result.get("tools", [])}
    independent = json.loads((out_dir / "independent.json").read_text(encoding="utf-8"))
    groups = load_name_map(ROOT / "labels" / "name_map.yaml")
    items = pool(system, independent["contacts"], cfg["checklist_iou"])
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    asks = [{"key": it["key"], "start_ms": it["start_ms"], "end_ms": it["end_ms"], "disagree": it["source"] != "both",
             "names": sorted({n for n in ((it["system"] or {}).get("object_label"), (it["gemini"] or {}).get("object")) if n})}
            for it in items]
    judge = GroqJudge(cfg, ROOT / "cache")
    answers = judge_items(judge, asks, lambda it, n: frames_for_item(cap, it, n, fps, cfg["groq_image_px"]), ROOT / "cache" / f"groq_{video.stem}.json")
    checklist = build_checklist(video.stem, variant, system, independent["contacts"], answers, cfg, groups,
                                {"models": {"independent": independent.get("model"), "judge": cfg["groq_model"]},
                                 "n_judged_by_groq": len(answers)}, status_of)
    out = out_dir / "checklist.json"
    out.write_text(json.dumps(checklist, indent=1), encoding="utf-8")
    sizes = {t: v for t, v in checklist["strata"].items()}
    print(f"{len(items)} pooled contacts ({sum(i['source'] == 'both' for i in items)} found by both, "
          f"{sum(i['source'] == 'system' for i in items)} only by the system, {sum(i['source'] == 'gemini_pro' for i in items)} only by Gemini Pro); "
          f"Groq judged {len(answers)}; check list {len(checklist['items'])} items, seed {cfg['checklist_seed']}; strata {sizes}")
    print(f"saved {out}")


if __name__ == "__main__":
    args, variant, rest = [], "ledger_sam", sys.argv[1:]
    while rest:
        a = rest.pop(0)
        if a == "--variant":
            variant = rest.pop(0)
        else:
            args.append(a)
    if len(args) != 1:
        sys.exit(__doc__)
    main(args[0], variant)
