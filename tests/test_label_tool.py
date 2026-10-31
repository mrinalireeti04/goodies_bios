import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dev"))

import label_tool as lt  # noqa: E402

ENTER, ESC, BS, TAB = lt.ENTER, lt.ESC, lt.BACKSPACE, lt.TAB


def session(tmp_path, fps=30.0):
    return lt.LabelSession("clip", fps, 3000, tmp_path / "labels" / "clip_contacts.csv")


def add(s, a=30, b=59, **kw):
    s.mark_start(a)
    s.mark_end(b)
    args = dict(hand="left", state="portable", interaction="hold", obj="Hammer", instance="Hammer1")
    args.update(kw)
    return s.add_contact(args.pop("hand"), args.pop("state"), args.pop("interaction"), args.pop("obj"), **args)


def read(tmp_path):
    with (tmp_path / "labels" / "clip_contacts.csv").open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_every_change_is_saved_at_once_and_names_are_normalised(tmp_path):
    s = session(tmp_path)
    row = add(s)
    assert row["start_ms"] == 1000 and row["end_ms"] == 2000       # the END frame (59) is the last one WITH contact
    saved = read(tmp_path)
    assert len(saved) == 1 and saved[0]["object"] == "hammer" and saved[0]["instance"] == "hammer1"
    assert list(saved[0]) == lt.COLUMNS


def test_resume_loads_rows_and_the_last_position(tmp_path):
    s = session(tmp_path)
    add(s)
    s.save(position=1234)
    again = session(tmp_path)
    assert len(again.rows) == 1 and again.position == 1234 and again.counts() == (1, 0)


def test_undo_removes_the_last_added_row_and_is_saved(tmp_path):
    s = session(tmp_path)
    add(s, 30, 59)
    add(s, 5, 9, obj="saw", instance="saw1")                       # earlier in time, but added last
    assert s.undo()["object"] == "saw"
    assert [r["object"] for r in read(tmp_path)] == ["hammer"]
    assert s.undo() is None                                         # only the latest addition can be undone
    assert s.undo() is None


def test_unsure_spans_open_close_and_are_saved_separately(tmp_path):
    s = session(tmp_path)
    assert s.toggle_unsure_span(300) is None and s.span_start == 300
    row = s.toggle_unsure_span(209)                                 # closing before the start is fine: order is fixed
    assert (row["start_ms"], row["end_ms"], row["kind"]) == (round(209 * 1000 / 30), round(300 * 1000 / 30 + 1000 / 30), "unsure_span")
    assert s.span_start is None and s.counts() == (0, 1)
    assert read(tmp_path)[0]["kind"] == "unsure_span"


def test_a_contact_needs_marks_valid_choices_and_a_name(tmp_path):
    s = session(tmp_path)
    with pytest.raises(ValueError):
        s.add_contact("left", "portable", "hold", "hammer")
    s.mark_start(10)
    s.mark_end(5)
    assert not s.ready()
    s.mark_end(10)                                                  # one-frame contact is allowed
    for bad in (dict(hand="middle"), dict(state="loose"), dict(interaction="grab"), dict(obj="  ")):
        s.mark_start(10), s.mark_end(10)
        with pytest.raises(ValueError):
            add(s, 10, 10, **bad)
    assert s.rows == []


def test_name_and_tag_suggestions_come_from_earlier_rows(tmp_path):
    s = session(tmp_path)
    add(s, 10, 20, obj="hammer", instance="hammer1")
    add(s, 40, 50, obj="drill", instance="drill1")
    add(s, 70, 80, obj="hammer", instance="hammer2")
    assert s.names() == ["hammer", "drill"] or s.names() == ["drill", "hammer"]
    assert s.tags_for("hammer")[0] in ("hammer2", "hammer1") and set(s.tags_for("hammer")) == {"hammer1", "hammer2"}
    assert s.tags_for("saw") == []


def keys(form, text_):
    for ch in text_:
        form.key(ord(ch) if ch != "\n" else ENTER)


def make_form(defaults=None, names=("hammer", "drill"), tags=None):
    base = {"hand": "right", "state": "portable", "interaction": "hold", "object": "", "unsure": False}
    base.update(defaults or {})
    tags = tags or {}
    return lt.Form(base, list(names), lambda obj: tags.get(obj, []))


def test_form_one_letter_answers_and_typed_names():
    f = make_form()
    keys(f, "lpt")                       # left, portable, touch
    keys(f, "chisel\n")
    keys(f, "chisel1\n")
    keys(f, "n")
    assert f.done and f.values == {"hand": "left", "state": "portable", "interaction": "touch", "object": "chisel",
                                   "instance": "chisel1", "unsure": False}


def test_form_enter_accepts_the_previous_answers():
    f = make_form({"hand": "left", "state": "fixed", "interaction": "touch", "object": "vise"}, tags={"vise": ["vise1"]})
    keys(f, "\n\n\n\n\n\n")
    assert f.done and f.values["hand"] == "left" and f.values["state"] == "fixed"
    assert f.values["object"] == "vise" and f.values["instance"] == "vise1" and f.values["unsure"] is False


def test_form_object_name_is_required_and_tab_completes_and_backspace_edits():
    f = make_form()
    keys(f, "rph")
    f.key(ENTER)                         # no default name and nothing typed: stays on the question
    assert f.i == 3 and not f.done
    keys(f, "dxx")
    f.key(BS), f.key(BS)
    f.key(TAB)                           # "d" completes to the earlier name "drill"
    assert f.buffer == "drill"
    f.key(ENTER)
    assert f.values["object"] == "drill"


def test_form_unsure_flag_and_escape():
    f = make_form({"object": "saw"})
    keys(f, "rph\n\n")
    f.key(ord("y"))
    assert f.done and f.values["unsure"] is True
    g = make_form()
    keys(g, "r")
    g.key(ESC)
    assert g.cancelled and not g.done


def test_typed_times():
    assert lt.TimeEntry.parse("75") == 75 and lt.TimeEntry.parse("1:15") == 75 and lt.TimeEntry.parse("75.5") == 75.5
    assert lt.TimeEntry.parse("1:02:03") is None and lt.TimeEntry.parse("abc") is None
