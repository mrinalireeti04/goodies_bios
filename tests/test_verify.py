import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dev"))

import verify_contacts as vc  # noqa: E402


def contact(n, start=1000, end=2000, hand="h_right", label="hammer"):
    return {"event_id": f"c_{n}", "start_ms": start * n, "end_ms": end * n + 1, "hand_id": hand, "object_id": f"obj_{n}",
            "object_label": label, "interaction": "hold"}


def session(tmp_path, n=6, kind="contact"):
    items = [contact(i) for i in range(1, n + 1)]
    return vc.VerifySession(kind, items, tmp_path / "labels" / "verify_clip.csv"), items


def test_answers_are_saved_at_once_and_the_run_is_resumable(tmp_path):
    s, items = session(tmp_path)
    s.record(items[0], "y")
    s.record(items[1], "n")
    again = vc.VerifySession("contact", items, tmp_path / "labels" / "verify_clip.csv")
    assert [it["event_id"] for it in again.pending()] == ["c_3", "c_4", "c_5", "c_6"]
    with (tmp_path / "labels" / "verify_clip.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["verdict"] for r in rows] == ["y", "n"] and rows[0]["object_label"] == "hammer"


def test_answering_the_same_item_again_replaces_the_answer(tmp_path):
    s, items = session(tmp_path)
    s.record(items[0], "y")
    s.record(items[0], "i")
    assert [r["verdict"] for r in s.rows] == ["i"]


def test_key_survives_renumbered_event_ids_but_not_a_changed_interval(tmp_path):
    s, items = session(tmp_path, n=2)
    s.record(items[0], "y")
    renumbered = [dict(items[0], event_id="c_99"), dict(items[1], start_ms=5)]
    again = vc.VerifySession("contact", renumbered, tmp_path / "labels" / "verify_clip.csv")
    assert [it["event_id"] for it in again.pending()] == ["c_2"]


def test_contact_statistics_as_counts(tmp_path):
    s, items = session(tmp_path, n=8)
    for item, v in zip(items, "yyyniioo"):
        s.record(item, v, "chisel" if v == "o" else "")
    st = s.stats()
    assert st["precision"] == (7, 8)      # 7 of 8 judged contacts were real
    assert st["naming"] == (5, 7)         # y,y,y,i,i have the right object; the two o's do not
    assert st["interaction"] == (5, 7)    # y,y,y,o,o have the right touch/hold; the two i's do not
    assert vc.format_stats(st).splitlines()[0] == "precision: 7/8 (88%)"


def test_wrong_object_needs_a_name_and_bad_verdicts_are_rejected(tmp_path):
    s, items = session(tmp_path)
    with pytest.raises(ValueError):
        s.record(items[0], "o")
    with pytest.raises(ValueError):
        s.record(items[0], "z")
    s.record(items[0], "o", " Chisel ")
    assert s.rows[0]["correct_name"] == "Chisel"


def test_safety_events_only_have_yes_or_no_and_a_precision(tmp_path):
    events = [{"event_id": f"s_{i}", "start_ms": i * 1000, "end_ms": i * 1000 + 500, "hand_id": "h_left",
               "hazard_object_id": "obj_5"} for i in range(1, 4)]
    s = vc.VerifySession("safety", events, tmp_path / "labels" / "verify_clip.csv")
    for e, v in zip(events, "yyn"):
        s.record(e, v)
    with pytest.raises(ValueError):
        s.record(events[0], "i")
    assert s.stats() == {"precision": (2, 3)} and s.rows[0]["object_id"] == "obj_5"


def test_contact_and_safety_answers_can_share_one_file_without_mixing(tmp_path):
    s, items = session(tmp_path, n=2)
    s.record(items[0], "y")
    other = vc.VerifySession("safety", [], tmp_path / "labels" / "verify_clip.csv")
    assert other.rows == []
    other.record({"event_id": "s_1", "start_ms": 1, "end_ms": 2, "hand_id": "h", "hazard_object_id": "obj_1"}, "n")
    assert len(vc.VerifySession("contact", items, tmp_path / "labels" / "verify_clip.csv").rows) == 1
