"""One command: side resolver -> gap filler -> validation, for every clip in out/ and out_rtm2/.

    python tools/postprocess_all.py                       # looks in out and out_rtm2
    python tools/postprocess_all.py out out_rtm2 --max-gap 2

For each <folder>/<clip>/hands.jsonl it writes hands_sided.jsonl (sides resolved) and
hands_final.jsonl (sides resolved + short gaps filled), checks both against the schema, and
prints a before/after table. Original files are never changed. Outputs stay in the git-ignored
result folders.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import fill_gaps as fg  # noqa: E402
import hand_detect as hd  # noqa: E402
import resolve_sides as rs  # noqa: E402


def write(recs, path):
    import json
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, allow_nan=False, separators=(",", ":")) + "\n")


def validate(path):
    n, err = hd.validate_file(str(path))
    return "OK" if err is None else f"FAIL: {err[:60]}"


def counts(recs):
    n = len(recs)
    with_hand = sum(1 for r in recs if r["hands"])
    unknown = sum(h["side"] == "unknown" for r in recs for h in r["hands"])
    interp = sum(h["source"] == "interpolated" for r in recs for h in r["hands"])
    return n, with_hand, unknown, interp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="*", default=["out", "out_rtm2"])
    ap.add_argument("--max-gap", type=int, default=2)
    args = ap.parse_args(argv)

    rows = []
    for folder in args.folders:
        for src in sorted(Path(folder).glob("*/hands.jsonl")):
            recs = fg.load(src)
            sided, _ = rs.resolve(recs)
            final, _ = fg.fill(sided, max_gap=args.max_gap)
            p1, p2 = src.with_name("hands_sided.jsonl"), src.with_name("hands_final.jsonl")
            write(sided, p1)
            write(final, p2)
            n, h0, u0, _ = counts(recs)
            _, h1, u1, i1 = counts(final)
            ok = "OK" if validate(p1) == "OK" and validate(p2) == "OK" else "FAIL"
            rows.append((f"{folder}/{src.parent.name}", n, h0, h1, u0, u1, i1, ok))

    if not rows:
        print("No hands.jsonl files found. Run from the BIOS folder.")
        return 1
    head = f"{'clip':<40}{'frames':>7}{'hand fr.':>16}{'unknown hands':>18}{'interp.':>9}  schema"
    print(head)
    print("-" * len(head))
    for name, n, h0, h1, u0, u1, i1, ok in rows:
        print(f"{name:<40}{n:>7}{f'{h0} -> {h1}':>16}{f'{u0} -> {u1}':>18}{i1:>9}  {ok}")
    print("\nhand fr. = frames with at least one hand; interp. = hands added by gap filling.")
    return 0 if all(r[-1] == "OK" for r in rows) else 2


if __name__ == "__main__":
    sys.exit(main())
