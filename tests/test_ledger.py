import json
import urllib.error
from pathlib import Path

import pytest

from ps2 import ledger, vlm

BASE = {"objects": {"workbench": {"type": "fixed", "prompts": ["workbench"]},
                    "table_saw": {"type": "fixed", "hazard": True, "prompts": ["table saw"]},
                    "hammer": {"type": "portable", "prompts": ["hammer"]}},
        "generic": {"tool": {"type": "portable", "prompts": ["tool"]}}, "ignore": ["hand"]}


def tool(key="hammer_1", name="Hammer", uses=(("00:10", "00:20"),), **kw):
    t = {"tool_key": key, "name": name, "synonyms": ["claw hammer"], "appearance": "red handle", "type": "portable",
         "powered_cutting_or_rotating": False, "workpiece": False,
         "uses": [{"start": a, "end": b, "hand": "left", "action": "hit", "certainty": "high"} for a, b in uses]}
    t.update(kw)
    return t


def answer(*tools, running=(), unsure=()):
    return {"tools": list(tools), "running": list(running), "unsure": list(unsure)}


def valid(data, duration=100.0, cfg=None, offset=0.0, length=None, cfg_over=None):
    return ledger.validate_ledger([(data, offset, length or duration)], duration, {"ledger_max_tools": 30, **(cfg_over or {})})


@pytest.mark.parametrize("text,expected", [("01:15", 75), ("1:02:03", 3723), ("75", 75), ("00:07.5", 7.5), (" 2:00 ", 120),
                                           ("abc", None), ("1:2:3:4", None), ("", None), (None, None), ("-1:00", None)])
def test_parse_time(text, expected):
    assert ledger.parse_time(text) == expected


def test_validation_normalises_names_and_converts_times_to_seconds():
    out = valid(answer(tool(name="  Claw  HAMMER ", key="Hammer 1")))
    t = out["tools"][0]
    assert t["tool_key"] == "hammer_1" and t["name"] == "claw hammer" and t["synonyms"] == ["claw hammer"] or t["synonyms"] == []
    assert t["uses"] == [{"start_s": 10, "end_s": 20, "hand": "left", "action": "hit", "certainty": "high"}]


def test_bad_intervals_and_malformed_entries_are_dropped():
    data = answer(
        tool(uses=(("00:10", "00:09"), ("00:30", "00:30"), ("02:00", "02:10"), ("01:35", "01:45"), ("01:40", "01:41"), ("xx", "00:05"))),
        {"tool_key": "x"}, {"name": "no key", "type": "portable"}, "garbage",
        running=[{"tool_key": "hammer_1", "start": "00:50", "end": "00:40"}, {"tool_key": "hammer_1", "start": "00:50", "end": "01:00"}],
        unsure=[{"description": "", "start": "00:01", "end": "00:02"}, {"description": "  a   thing ", "start": "00:01", "end": "00:02"}])
    out = valid(data, duration=100.0)
    assert out["tools"][0]["uses"] == []               # end<=start, zero length, outside the video, unreadable: all dropped
    assert len(out["tools"]) == 1                     # entries without a key / name / type are gone
    assert out["running"] == [{"tool_key": "hammer_1", "start_s": 50, "end_s": 60}]
    assert out["unsure"] == [{"description": "a thing", "start_s": 1, "end_s": 2}]


def test_end_slightly_past_the_video_is_clamped_but_far_past_is_dropped():
    out = valid(answer(tool(uses=(("01:30", "01:41"), ("01:35", "01:50")))), duration=100.0)
    assert [(u["start_s"], u["end_s"]) for u in out["tools"][0]["uses"]] == [(90, 100)]


def test_same_tool_key_in_several_windows_is_merged_and_later_windows_are_offset():
    first = answer(tool(uses=(("00:10", "00:20"),)))
    second = answer(tool(uses=(("00:05", "00:15"),), synonyms=["mallet"]))
    out = ledger.validate_ledger([(first, 0.0, 300.0), (second, 300.0, 300.0)], 600.0, {"ledger_max_tools": 30})
    assert len(out["tools"]) == 1
    assert [(u["start_s"], u["end_s"]) for u in out["tools"][0]["uses"]] == [(10, 20), (305, 315)]
    absolute = answer(tool(uses=(("05:10", "05:20"),)))        # a window answered with the original clock
    out = ledger.validate_ledger([(absolute, 300.0, 300.0)], 600.0, {"ledger_max_tools": 30})
    assert [(u["start_s"], u["end_s"]) for u in out["tools"][0]["uses"]] == [(310, 320)]


def test_tool_cap_keeps_the_most_used_tools_and_their_running_intervals():
    tools = [tool(key=f"t{i}", name=f"tool {i}", uses=((f"00:{10 + i:02d}", f"00:{10 + i + i + 1:02d}"),)) for i in range(5)]
    out = valid(answer(*tools, running=[{"tool_key": "t0", "start": "00:00", "end": "00:05"}, {"tool_key": "t4", "start": "00:00", "end": "00:05"}]),
                cfg_over={"ledger_max_tools": 2})
    assert [t["tool_key"] for t in out["tools"]] == ["t4", "t3"] and [r["tool_key"] for r in out["running"]] == ["t4"]


def test_long_videos_are_split_into_windows():
    assert ledger.windows(250, 300) == [(0.0, 250)] and ledger.windows(300, 300) == [(0.0, 300)]
    assert ledger.windows(650, 300) == [(0.0, 300), (300.0, 600.0), (600.0, 650)]


def test_vocabulary_has_ledger_tools_synonyms_surfaces_but_no_workpieces_and_only_running_machines_are_hazards():
    data = valid(answer(
        tool(key="hammer_1", name="Hammer"),
        tool(key="board_1", name="plywood", workpiece=True),
        tool(key="drill_1", name="power drill", powered_cutting_or_rotating=True),
        tool(key="saw_1", name="band saw", type="fixed", powered_cutting_or_rotating=True),
        tool(key="press_1", name="drill press", type="fixed", powered_cutting_or_rotating=True),
        running=[{"tool_key": "saw_1", "start": "00:10", "end": "00:20"}]))
    v = ledger.ledger_vocab(data, BASE)
    assert set(v["objects"]) == {"hammer", "power_drill", "band_saw", "drill_press", "workbench"}   # no plywood, no base hammer/table_saw
    assert v["objects"]["hammer"]["prompts"] == ["hammer", "claw hammer"]
    assert v["objects"]["power_drill"]["hazard"] is True          # portable and powered
    assert v["objects"]["band_saw"]["hazard"] is True             # fixed and in the running list
    assert "hazard" not in v["objects"]["drill_press"]            # fixed, powered, never seen running
    assert v["generic"] == BASE["generic"] and v["ignore"] == BASE["ignore"]


# ---------------------------------------------------------------- Files API and the whole flow, with a fake network

class FakeHttp:
    def __init__(self):
        self.calls, self.deleted, self.state = [], [], ["ACTIVE"] * 20

    def __call__(self, method, url, headers, data, timeout):
        self.calls.append((method, url, dict(headers)))
        if method == "POST" and url.endswith("/upload/v1beta/files"):
            assert headers["X-Goog-Upload-Command"] == "start" and headers["X-Goog-Upload-Protocol"] == "resumable"
            return {"x-goog-upload-url": "https://upload.example/session1"}, b""
        if method == "POST" and url == "https://upload.example/session1":
            assert headers["X-Goog-Upload-Command"] == "upload, finalize" and data
            return {}, json.dumps({"file": {"name": "files/abc", "uri": "https://files/abc", "state": "PROCESSING"}}).encode()
        if method == "GET":
            return {}, json.dumps({"state": self.state.pop(0), "uri": "https://files/abc"}).encode()
        if method == "DELETE":
            self.deleted.append(url)
            return {}, b"{}"
        raise AssertionError((method, url))


class FakeGen:
    """Stands in for vlm.Gemini.generate."""

    def __init__(self, cfg, text, available=True):
        self.cfg, self.text, self.available, self.calls, self.requests, self.key = cfg, text, available, 0, [], "k"

    def generate(self, parts, max_tokens, schema=None, timeout=None):
        self.calls += 1
        self.requests.append((parts, schema, timeout))
        return self.text


def make_video(tmp_path):
    p = tmp_path / "clip.mp4"
    if not p.exists():
        p.write_bytes(b"not really a video")
    return p


def test_upload_polls_until_active_and_delete_is_reported(tmp_path):
    http = FakeHttp()
    http.state = ["PROCESSING", "ACTIVE"]
    api = ledger.FilesAPI("k", http)
    info = api.upload(make_video(tmp_path), "video/mp4", "ps2-x")
    assert info["name"] == "files/abc"
    assert api.wait_active("files/abc", sleep=lambda s: None)["state"] == "ACTIVE"
    assert api.delete("files/abc") is True and http.deleted


def test_a_failed_upload_state_stops_the_wait(tmp_path):
    http = FakeHttp()
    http.state = ["FAILED"]
    with pytest.raises(RuntimeError):
        ledger.FilesAPI("k", http).wait_active("files/abc", sleep=lambda s: None)


def build(tmp_path, cfg, text, duration=100.0, http=None, copy=None, available=True):
    http = http or FakeHttp()
    gem = FakeGen(cfg, text, available)
    video = make_video(tmp_path)
    result = ledger.build_ledger(video, duration, gem, cfg, "clip", tmp_path / "cache", tmp_path / "tmp",
                                 ledger.FilesAPI("k", http), copy)
    return result, gem, http


def test_flow_sends_the_file_with_fps_and_schema_then_deletes_the_upload_and_the_copy(tmp_path, cfg):
    def make_copy(video, out_dir, height):
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / "copy.mp4"
        p.write_bytes(b"small")
        return p

    (data, info), gem, http = build(tmp_path, cfg, json.dumps(answer(tool())), copy=make_copy)
    assert data["tools"][0]["tool_key"] == "hammer_1"
    parts, schema_used, timeout = gem.requests[0]
    assert parts[0]["file_data"]["file_uri"] == "https://files/abc" and parts[0]["video_metadata"] == {"fps": cfg["ledger_fps"]}
    assert "do not infer tools from context" in parts[1]["text"] and schema_used is ledger.SCHEMA and timeout == cfg["ledger_timeout_s"]
    assert info["upload_deleted"] is True and info["temp_deleted"] is True and http.deleted
    assert not (tmp_path / "tmp" / "copy.mp4").exists()
    assert (tmp_path / "cache" / "ledger_clip.json").exists()


def test_long_video_is_clipped_with_offsets_not_prompt_text(tmp_path, cfg):
    (data, info), gem, _ = build(tmp_path, {**cfg, "ledger_max_window_s": 60}, json.dumps(answer(tool(uses=(("00:05", "00:15"),)))),
                                 duration=130.0, copy=None)
    metas = [req[0][0]["video_metadata"] for req in gem.requests]
    assert metas == [{"fps": 2, "start_offset": "0s", "end_offset": "60s"}, {"fps": 2, "start_offset": "60s", "end_offset": "120s"},
                     {"fps": 2, "start_offset": "120s", "end_offset": "130s"}]
    assert info["windows"] == 3 and len(data["tools"][0]["uses"]) == 3
    assert "start_offset" not in gem.requests[0][0][1]["text"]


def test_cache_is_used_for_the_same_file_and_not_for_a_changed_one(tmp_path, cfg):
    (first, _), _, _ = build(tmp_path, cfg, json.dumps(answer(tool())))
    http = FakeHttp()
    (again, info), gem, _ = build(tmp_path, cfg, "unused", http=http)
    assert again == first and info["from_cache"] and gem.calls == 0 and not http.calls
    (tmp_path / "clip.mp4").write_bytes(b"a different, longer file")
    (changed, info2), gem2, _ = build(tmp_path, cfg, json.dumps(answer(tool(name="saw"))))
    assert not info2["from_cache"] and gem2.calls == 1 and changed["tools"][0]["name"] == "saw"


def test_the_upload_is_deleted_even_when_the_request_fails(tmp_path, cfg):
    class Boom(FakeGen):
        def generate(self, *a, **k):
            raise RuntimeError("boom")

    http = FakeHttp()
    gem = Boom(cfg, "")
    data, info = ledger.build_ledger(make_video(tmp_path), 100.0, gem, cfg, "clip", tmp_path / "cache", tmp_path / "tmp",
                                     ledger.FilesAPI("k", http), None)
    assert data is None and "boom" in info["error"] and info["upload_deleted"] is True and http.deleted


def test_no_key_or_empty_or_invalid_answers_give_no_ledger_and_no_upload(tmp_path, cfg):
    (data, info), _, http = build(tmp_path, cfg, "x", available=False)
    assert data is None and not http.calls and "key" in info["error"]
    (data, info), _, http = build(tmp_path, cfg, json.dumps(answer()))
    assert data is None and info["error"] == "the ledger came back empty" and http.deleted
    (data, info), _, _ = build(tmp_path, cfg, "not json at all")
    assert data is None and info["error"]


def test_background_job_returns_the_same_result(tmp_path, cfg):
    gem = FakeGen(cfg, json.dumps(answer(tool())))
    job = ledger.LedgerJob(make_video(tmp_path), 100.0, gem, cfg, "clip", tmp_path / "cache", tmp_path / "tmp",
                           files=ledger.FilesAPI("k", FakeHttp()), make_copy=None)
    data, info = job.result()
    assert data["tools"][0]["name"] == "hammer" and info["upload_deleted"] is True
