"""Evaluate the system from YOUR answers on the check list (dev only): "AI-proposed, human-verified".

Usage: python dev/evaluate.py <clip_stem> [--variant ledger_sam] [--window START END] [--seed N]

Reads outputs/<clip>/checklist.json (dev/checklist.py), your answers in labels/verify_<clip>.csv (dev/verify_contacts.py
--checklist) and, optionally, labels/<clip>_contacts.csv (dev/label_tool.py). Writes outputs/eval/<clip>_<variant>.md/.json.

How the numbers are made (each metric says where it comes from):
- The check list is NOT a random sample of all contacts: disagreements are over-represented. So every estimate is
  STRATIFIED by tier (tier 1 = complete disagreement, 2 = other disagreement, 3 = unanimous) using the tier sizes in the
  pooled population, and the 95% confidence interval is a bootstrap within tiers (fixed seed). With ~40 answers these
  intervals are wide; read them.
- Recall is only recall of the POOLED real contacts: contacts that no model proposed are not counted at all.
- Agreement between models is agreement, NOT accuracy: two models can agree and both be wrong.
- Exact recall and timing need your own labels for a window; without them that part is skipped.
"""
import csv
import json
import random
import statistics
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from checklist import hands_compatible, hand_of, interval_iou, load_name_map, names_match  # noqa: E402

REAL = {"y", "o", "i"}
BOOTSTRAP = 2000


# ---------------------------------------------------------------- stratified estimates

def _ratio(sample: dict[str, list[dict]], population: dict[str, float], num, den) -> float | None:
    """sum_h N_h * num_h / sum_h N_h * den_h, where num_h / den_h are the mean of num/den over tier h's answers."""
    top = bottom = 0.0
    for tier, rows in sample.items():
        if not rows:
            continue
        top += population.get(tier, 0) * sum(num(r) for r in rows) / len(rows)
        bottom += population.get(tier, 0) * sum(den(r) for r in rows) / len(rows)
    return top / bottom if bottom > 0 else None


def estimate(sample: dict[str, list[dict]], population: dict[str, float], num, den, seed: int = 7) -> dict:
    """Point estimate and a bootstrap 95% interval (resampling the answers inside every tier)."""
    point = _ratio(sample, population, num, den)
    rng = random.Random(seed)
    draws = []
    for _ in range(BOOTSTRAP):
        resample = {t: [rng.choice(rows) for _ in rows] for t, rows in sample.items()}
        value = _ratio(resample, population, num, den)
        if value is not None:
            draws.append(value)
    draws.sort()
    n = sum(len(r) for r in sample.values())
    if point is None or not draws:
        return {"value": None, "low": None, "high": None, "n": n}
    return {"value": point, "low": draws[int(0.025 * len(draws))], "high": draws[min(int(0.975 * len(draws)), len(draws) - 1)], "n": n}


# ---------------------------------------------------------------- the answers

def load_answers(csv_path: Path) -> dict[str, dict]:
    if not csv_path.exists():
        return {}
    with csv_path.open(encoding="utf-8", newline="") as f:
        return {r["key"].removeprefix("check|"): r for r in csv.DictReader(f) if r["kind"] == "check"}


def answered_sample(checklist: dict, answers: dict[str, dict]) -> dict[str, list[dict]]:
    """tier -> answered items: {source, verdict, identified, flagged}."""
    population = {p["key"]: p for p in checklist["population"]}
    sample: dict[str, list[dict]] = {"1": [], "2": [], "3": []}
    for item in checklist["items"]:
        a = answers.get(item["key"])
        if not a or a["verdict"] not in ("y", "n", "o", "i"):
            continue
        pop = population[item["key"]]
        sample[str(item["tier"])].append({"source": item["source"], "verdict": a["verdict"], "key": item["key"],
                                          "identified": bool(pop.get("identified")), "flagged": bool(pop.get("flagged"))})
    return sample


def tier_sizes(checklist: dict, keep) -> dict[str, float]:
    sizes: dict[str, float] = {"1": 0, "2": 0, "3": 0}
    for p in checklist["population"]:
        if keep(p):
            sizes[str(p["tier"])] += 1
    return sizes


def only(sample: dict[str, list[dict]], keep) -> dict[str, list[dict]]:
    return {t: [r for r in rows if keep(r)] for t, rows in sample.items()}


SYSTEM = lambda x: x["source"] in ("system", "both")  # noqa: E731
GEMINI = lambda x: x["source"] in ("gemini_pro", "both")  # noqa: E731


def human_metrics(checklist: dict, answers: dict[str, dict], seed: int) -> dict:
    """Everything that comes from your answers. Each entry says what it is."""
    sample = answered_sample(checklist, answers)
    n_all = sum(len(r) for r in sample.values())
    out = {"answered": n_all, "of": len(checklist["items"]), "source": "your answers on the check list, stratified by tier"}
    if n_all == 0:
        return out
    sys_pop = tier_sizes(checklist, SYSTEM)
    sys_sample = only(sample, SYSTEM)
    real = lambda r: 1.0 if r["verdict"] in REAL else 0.0  # noqa: E731
    one = lambda r: 1.0  # noqa: E731
    out["precision"] = estimate(sys_sample, sys_pop, real, one, seed)
    # naming among real system contacts: weight each tier by the estimated number of real contacts in it
    real_pop = {t: sys_pop[t] * (sum(real(r) for r in sys_sample[t]) / len(sys_sample[t]) if sys_sample[t] else 0) for t in sys_pop}
    real_sample = only(sys_sample, lambda r: r["verdict"] in REAL)
    named = lambda r: 1.0 if r["verdict"] in ("y", "i") else 0.0  # noqa: E731
    out["naming_accuracy"] = estimate(real_sample, real_pop, named, one, seed)
    out["identified_precision"] = estimate(only(real_sample, lambda r: r["identified"]), real_pop, named, one, seed)
    # share of wrong names that the system flagged (unidentified, or in the review queue)
    wrong_pop = {t: sys_pop[t] * (sum(r["verdict"] == "o" for r in sys_sample[t]) / len(sys_sample[t]) if sys_sample[t] else 0) for t in sys_pop}
    wrong_sample = only(sys_sample, lambda r: r["verdict"] == "o")
    out["wrong_names_flagged"] = estimate(wrong_sample, wrong_pop, lambda r: 1.0 if (r["flagged"] or not r["identified"]) else 0.0, one, seed)
    interaction_pop = real_pop
    out["interaction_accuracy"] = estimate(real_sample, interaction_pop, lambda r: 1.0 if r["verdict"] in ("y", "o") else 0.0, one, seed)
    # recall of the pooled real contacts (those no model proposed are not counted)
    all_pop = tier_sizes(checklist, lambda p: True)
    out["recall_of_pooled_real"] = estimate(sample, all_pop, lambda r: 1.0 if (r["verdict"] in REAL and SYSTEM(r)) else 0.0,
                                            lambda r: 1.0 if r["verdict"] in REAL else 0.0, seed)
    out["gemini_pro_recall_of_pooled_real"] = estimate(sample, all_pop, lambda r: 1.0 if (r["verdict"] in REAL and GEMINI(r)) else 0.0,
                                                       lambda r: 1.0 if r["verdict"] in REAL else 0.0, seed)
    gem_sample = only(sample, GEMINI)
    out["gemini_pro_precision"] = estimate(gem_sample, tier_sizes(checklist, GEMINI), real, one, seed)
    out["counts"] = {v: sum(r["verdict"] == v for rows in sample.values() for r in rows) for v in "yoin"}
    out["missing_tiers"] = [t for t, rows in sample.items() if not rows and checklist["strata"].get(t, {}).get("N")]
    return out


# ---------------------------------------------------------------- agreement (no human involved)

def agreement(checklist: dict, groups) -> dict:
    """How often the models give the same vote over ALL pooled contacts. Agreement is not accuracy."""
    pairs = {"system_vs_gemini_pro": ("system", "gemini"), "system_vs_groq": ("system", "groq"), "gemini_pro_vs_groq": ("gemini", "groq")}
    out = {"source": "all pooled items, no human involved; AGREEMENT, not accuracy", "items": len(checklist["population"])}
    for name, (a, b) in pairs.items():
        same = total = skipped = names_same = names_total = 0
        for p in checklist["population"]:
            va, vb = p["votes"][a], p["votes"][b]
            if va in (None, "unsure") or vb in (None, "unsure"):
                skipped += 1
                continue
            total += 1
            same += va == vb
            if va == vb == "yes" and p["votes"]["names"].get(a) and p["votes"]["names"].get(b):
                names_total += 1
                names_same += names_match(p["votes"]["names"][a], p["votes"]["names"][b], groups)
        out[name] = {"agree": same, "of": total, "skipped_unsure_or_unjudged": skipped,
                     "same_tool_name_when_both_say_contact": f"{names_same}/{names_total}"}
    return out


# ---------------------------------------------------------------- exact recall and timing for a labelled window

def read_labels(path: Path) -> tuple[list[dict], list[dict]]:
    if not path.exists():
        return [], []
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    contacts = [dict(r, start_ms=int(r["start_ms"]), end_ms=int(r["end_ms"])) for r in rows if r["kind"] == "contact"]
    spans = [dict(r, start_ms=int(r["start_ms"]), end_ms=int(r["end_ms"])) for r in rows if r["kind"] == "unsure_span"]
    return contacts, spans


def match_exact(labels: list[dict], preds: list[dict], iou_min: float, use_hand: bool) -> list[tuple[int, int, float]]:
    pairs = sorted(((interval_iou(l, p), i, j) for i, l in enumerate(labels) for j, p in enumerate(preds)
                    if not use_hand or hands_compatible(l["hand"], hand_of(p["hand_id"]))), reverse=True)
    used_l, used_p, out = set(), set(), []
    for score, i, j in pairs:
        if score < iou_min:
            break
        if i not in used_l and j not in used_p:
            used_l.add(i)
            used_p.add(j)
            out.append((i, j, score))
    return out


def window_metrics(labels: list[dict], spans: list[dict], system: list[dict], window: tuple[int, int] | None, groups) -> dict | None:
    """Recall, precision and timing of the system against YOUR labels inside a window (unsure labels and spans excluded)."""
    if not labels:
        return None
    lo, hi = window or (min(r["start_ms"] for r in labels + spans), max(r["end_ms"] for r in labels + spans))
    inside = lambda r: r["end_ms"] > lo and r["start_ms"] < hi  # noqa: E731
    unsure = lambda r: any(interval_iou(r, s) > 0 or (s["start_ms"] <= r["start_ms"] < s["end_ms"]) for s in spans)  # noqa: E731
    keep_labels = [l for l in labels if inside(l) and str(l.get("unsure")) != "1" and not unsure(l)]
    keep_preds = [p for p in system if inside(p) and not unsure(p)]
    out = {"source": "your labels (labels/<clip>_contacts.csv) inside the window", "window_ms": [lo, hi],
           "labelled": len(keep_labels), "predicted": len(keep_preds)}
    for iou_min in (0.5, 0.3):
        for use_hand in (True, False):
            m = match_exact(keep_labels, keep_preds, iou_min, use_hand)
            tag = f"iou{iou_min}_{'hand' if use_hand else 'anyhand'}"
            out[tag] = {"recall": f"{len(m)}/{len(keep_labels)}", "precision": f"{len(m)}/{len(keep_preds)}",
                        "f1": round(2 * len(m) / (len(keep_labels) + len(keep_preds)), 3) if keep_labels or keep_preds else None}
    m = match_exact(keep_labels, keep_preds, 0.3, False)
    if m:
        starts = sorted(abs(keep_preds[j]["start_ms"] - keep_labels[i]["start_ms"]) for i, j, _ in m)
        ends = sorted(abs(keep_preds[j]["end_ms"] - keep_labels[i]["end_ms"]) for i, j, _ in m)
        p90 = lambda xs: xs[min(int(0.9 * len(xs)), len(xs) - 1)]  # noqa: E731
        out["timing_ms_at_iou0.3"] = {"matched": len(m), "start_median": statistics.median(starts), "start_p90": p90(starts),
                                      "end_median": statistics.median(ends), "end_p90": p90(ends)}
        out["name_correct"] = f"{sum(names_match(keep_labels[i]['object'], keep_preds[j].get('object_label') or '', groups) for i, j, _ in m)}/{len(m)}"
        out["touch_hold_correct"] = f"{sum(keep_labels[i]['interaction'] == keep_preds[j]['interaction'] for i, j, _ in m)}/{len(m)}"
        out["state_correct"] = f"{sum(keep_labels[i]['state'] == keep_preds[j]['state'] for i, j, _ in m)}/{len(m)}"
    return out


# ---------------------------------------------------------------- report

def pct(e: dict | None) -> str:
    if not e or e.get("value") is None:
        return "n/a (no answers for this)"
    return f"{100 * e['value']:.0f}% (95% CI {100 * e['low']:.0f}-{100 * e['high']:.0f}%, from {e['n']} answers)"


def report(clip: str, variant: str, human: dict, agree: dict, window: dict | None, checklist: dict) -> str:
    lines = [f"# Evaluation: {clip} ({variant})", "",
             f"Check list: {human.get('of', len(checklist['items']))} items from {checklist['n_pooled']} pooled contacts; "
             f"seed {checklist['seed']}; strata (N pooled / n listed): "
             + ", ".join(f"tier {t}: {v['N']}/{v['n']}" for t, v in checklist["strata"].items()),
             f"You answered {human.get('answered', 0)} items. Source of every number is stated in its row.", ""]
    if human.get("answered"):
        c = human["counts"]
        lines += ["## From your answers", f"_({human['source']}; bootstrap CIs are wide with so few answers)_", "",
                  "| Metric | Estimate | What it is |", "|---|---|---|",
                  f"| Precision of system contacts | {pct(human['precision'])} | real contacts among the system's contacts |",
                  f"| Naming accuracy | {pct(human['naming_accuracy'])} | right tool among real system contacts |",
                  f"| Precision of 'identified' labels | {pct(human['identified_precision'])} | right tool among real contacts the system called identified |",
                  f"| Wrong names that were flagged | {pct(human['wrong_names_flagged'])} | wrong names marked unidentified or put in review |",
                  f"| Touch/hold accuracy | {pct(human['interaction_accuracy'])} | right interaction among real system contacts |",
                  f"| Estimated recall (pooled real contacts) | {pct(human['recall_of_pooled_real'])} | **misses by every model are not counted** |",
                  f"| Gemini Pro recall (same pool) | {pct(human['gemini_pro_recall_of_pooled_real'])} | for comparison |",
                  f"| Gemini Pro precision | {pct(human['gemini_pro_precision'])} | for comparison |", "",
                  f"Your answers: {c['y']} correct, {c['o']} wrong object, {c['i']} wrong touch/hold, {c['n']} no real contact."]
        if human["missing_tiers"]:
            lines.append(f"Tiers without answers (estimates leave them out): {', '.join(human['missing_tiers'])}.")
        lines.append("")
    lines += ["## Agreement between models (no human; AGREEMENT, not accuracy)", "", "| Pair | Same vote | Same tool name when both see a contact | Skipped |", "|---|---|---|---|"]
    for pair in ("system_vs_gemini_pro", "system_vs_groq", "gemini_pro_vs_groq"):
        a = agree[pair]
        lines.append(f"| {pair.replace('_', ' ')} | {a['agree']}/{a['of']} | {a['same_tool_name_when_both_say_contact']} | {a['skipped_unsure_or_unjudged']} |")
    if window:
        lines += ["", "## Exact recall and timing in your labelled window", f"_({window['source']}: {window['labelled']} labels, {window['predicted']} predictions)_", "",
                  "| Matching | Recall | Precision | F1 |", "|---|---|---|---|"]
        for tag in ("iou0.5_hand", "iou0.5_anyhand", "iou0.3_hand", "iou0.3_anyhand"):
            lines.append(f"| {tag} | {window[tag]['recall']} | {window[tag]['precision']} | {window[tag]['f1']} |")
        if "timing_ms_at_iou0.3" in window:
            t = window["timing_ms_at_iou0.3"]
            lines += ["", f"Timing at IoU 0.3 on {t['matched']} matches: start error median {t['start_median']} ms / p90 {t['start_p90']} ms; "
                      f"end error median {t['end_median']} ms / p90 {t['end_p90']} ms. Name correct {window['name_correct']}, "
                      f"touch/hold correct {window['touch_hold_correct']}, state correct {window['state_correct']}."]
    return "\n".join(lines) + "\n"


def main(clip: str, variant: str | None, window: tuple[int, int] | None, seed: int) -> None:
    out_dir = ROOT / "outputs" / clip
    checklist = json.loads((out_dir / "checklist.json").read_text(encoding="utf-8"))
    variant = variant or checklist["variant"]
    groups = load_name_map(ROOT / "labels" / "name_map.yaml")
    answers = load_answers(ROOT / "labels" / f"verify_{clip}.csv")
    human = human_metrics(checklist, answers, seed)
    agree = agreement(checklist, groups)
    system = json.loads((out_dir / f"{variant}.json").read_text(encoding="utf-8"))["contacts"]
    labels, spans = read_labels(ROOT / "labels" / f"{clip}_contacts.csv")
    win = window_metrics(labels, spans, system, window, groups)
    text = report(clip, variant, human, agree, win, checklist)
    target = ROOT / "outputs" / "eval"
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{clip}_{variant}.md").write_text(text, encoding="utf-8")
    (target / f"{clip}_{variant}.json").write_text(json.dumps({"human": human, "agreement": agree, "window": win}, indent=1), encoding="utf-8")
    print(text)
    print(f"saved to {target / (clip + '_' + variant + '.md')} and .json")


if __name__ == "__main__":
    argv = sys.argv[1:]
    variant = argv[argv.index("--variant") + 1] if "--variant" in argv else None
    win = (int(float(argv[argv.index("--window") + 1]) * 1000), int(float(argv[argv.index("--window") + 2]) * 1000)) if "--window" in argv else None
    seed = int(argv[argv.index("--seed") + 1]) if "--seed" in argv else 7
    skip = {"--variant", "--seed"}
    positional = [a for k, a in enumerate(argv) if not a.startswith("--") and not (k > 0 and argv[k - 1] in skip)
                  and not (k > 0 and argv[k - 1] == "--window") and not (k > 1 and argv[k - 2] == "--window")]
    if len(positional) != 1:
        sys.exit(__doc__)
    main(positional[0], variant, win, seed)
