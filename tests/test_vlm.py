import json
import urllib.error

import numpy as np
import pytest

from ps2 import summary, vlm

LABELS = ["hammer", "power_drill", "angle_grinder"]
VOCAB = {"objects": {"hammer": {"type": "portable", "prompts": ["hammer"]},
                     "power_drill": {"type": "portable", "prompts": ["power drill"]},
                     "angle_grinder": {"type": "portable", "hazard": True, "prompts": ["angle grinder"]}},
         "generic": {"tool": {"type": "portable", "prompts": ["tool"]}}, "ignore": []}


def reply(text, finish="STOP"):
    return {"candidates": [{"finishReason": finish, "content": {"parts": [{"text": text}]}}]}


def http_error(code):
    return urllib.error.HTTPError("u", code, "x", {}, None)


class FakePost:
    """Plays back texts (or exceptions) one per request and records every request."""

    def __init__(self, *items):
        self.items, self.calls = list(items), []

    def __call__(self, url, key, body, timeout):
        self.calls.append((url, key, body, timeout))
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item if isinstance(item, dict) else reply(item)


def client(cfg, monkeypatch, fake, key="test-key", **overrides):
    monkeypatch.setenv("GEMINI_API_KEY", key) if key else monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    waits = []
    gem = vlm.Gemini({**cfg, **overrides}, post=fake, sleep=waits.append)
    gem.waits = waits
    return gem


def crop(value=100):
    return np.full((32, 32, 3), value, np.uint8)


def ask(gem, cfg, tmp_path, crops=None):
    return vlm.name_objects(gem, crops or {"obj_1": crop()}, LABELS, "clip", cfg, tmp_path / "cache")


@pytest.mark.parametrize("text,expected", [
    ("power_drill", ("label", "power_drill")), ("Power Drill.", ("label", "power_drill")),
    ('"hammer"', ("label", "hammer")), ("other: Chisel ", ("other", "chisel")),
    ("unknown", ("unknown", "")), ("banana_peeler", ("unknown", "")), ("other:", ("unknown", "")),
])
def test_parse_answer(text, expected):
    assert vlm.parse_answer(text, LABELS) == expected


def test_no_key_means_no_call_and_no_answers(tmp_path, cfg, monkeypatch):
    fake = FakePost("hammer")
    gem = client(cfg, monkeypatch, fake, key=None)
    assert not gem.available and ask(gem, cfg, tmp_path) == {} and fake.calls == []


def test_request_is_inline_with_the_label_list_and_model_settings(tmp_path, cfg, monkeypatch):
    fake = FakePost("hammer")
    gem = client(cfg, monkeypatch, fake)
    assert ask(gem, cfg, tmp_path) == {"obj_1": ("label", "hammer")}
    url, key, body, timeout = fake.calls[0]
    parts, gen = body["contents"][0]["parts"], body["generationConfig"]
    assert cfg["vlm_model"] in url and key == "test-key" and timeout == cfg["vlm_timeout_s"]
    assert "hammer, power_drill, angle_grinder" in parts[0]["text"] and "other:<short name>" in parts[0]["text"]
    assert parts[1]["inline_data"]["mime_type"] == "image/jpeg"
    assert gen["temperature"] == 0 and gen["thinkingConfig"] == {"thinkingLevel": cfg["vlm_thinking_level"]}
    assert not any("test-key" in f.read_text(encoding="utf-8") for f in (tmp_path / "cache").iterdir())


def test_answers_are_cached_per_clip_and_object_and_checked_against_the_crop(tmp_path, cfg, monkeypatch):
    ask(client(cfg, monkeypatch, FakePost("hammer")), cfg, tmp_path)
    assert (tmp_path / "cache" / "clip_obj_1.json").exists()
    again = FakePost("power_drill")
    gem = client(cfg, monkeypatch, again)
    assert ask(gem, cfg, tmp_path) == {"obj_1": ("label", "hammer")} and again.calls == [] and gem.calls == 0
    changed = client(cfg, monkeypatch, FakePost("power_drill"))   # same id but a different crop: ask again
    assert ask(changed, cfg, tmp_path, {"obj_1": crop(200)}) == {"obj_1": ("label", "power_drill")}


def test_rate_limit_is_retried_with_exponential_backoff(cfg, monkeypatch):
    fake = FakePost(http_error(429), http_error(503), "hammer")
    gem = client(cfg, monkeypatch, fake)
    assert gem.generate([{"text": "x"}], 10) == "hammer"
    assert gem.calls == 3 and gem.waits == [cfg["vlm_backoff_s"], cfg["vlm_backoff_s"] * 2]


def test_gives_up_after_three_tries_then_one_try_on_the_fallback_model(cfg, monkeypatch):
    fake = FakePost(http_error(429), http_error(429), http_error(429), "hammer")
    gem = client(cfg, monkeypatch, fake)
    assert gem.generate([{"text": "x"}], 10) == "hammer"
    assert [c[0].split("models/")[1].split(":")[0] for c in fake.calls] == \
        [cfg["vlm_model"]] * 3 + [cfg["vlm_fallback_model"]]
    assert "thinkingConfig" not in fake.calls[3][2]["generationConfig"]   # main model's setting only
    dead = client(cfg, monkeypatch, FakePost(*[http_error(503)] * 4))
    assert dead.generate([{"text": "x"}], 10) is None and dead.available   # a bad moment is not a permanent failure


def test_call_limit_per_clip_counts_every_request(tmp_path, cfg, monkeypatch):
    gem = client(cfg, monkeypatch, FakePost(*["hammer"] * 5), vlm_max_calls_per_clip=2)
    crops = {f"obj_{i}": crop(i * 20) for i in range(1, 5)}
    assert set(ask(gem, cfg, tmp_path, crops)) == {"obj_1", "obj_2"} and gem.calls == 2


def test_bad_key_or_model_stops_asking_and_never_raises(tmp_path, cfg, monkeypatch):
    fake = FakePost(http_error(403), "hammer")
    gem = client(cfg, monkeypatch, fake)
    assert ask(gem, cfg, tmp_path, {"obj_1": crop(10), "obj_2": crop(20)}) == {} and len(fake.calls) == 1
    assert not gem.available


def test_cut_off_or_odd_replies_are_skipped(cfg, monkeypatch):
    gem = client(cfg, monkeypatch, FakePost(reply("Saf", "MAX_TOKENS"), {"unexpected": 1}, TimeoutError(), TimeoutError(),
                                            TimeoutError(), TimeoutError()))
    assert gem.generate([{"text": "x"}], 10) is None            # cut off by the token limit
    assert gem.generate([{"text": "x"}], 10) is None            # reply without candidates
    assert gem.generate([{"text": "x"}], 10) is None            # timeouts, then the fallback also times out


def det(oid="obj_1", label="tool", box=(0, 0, 100, 100)):
    return {"object_id": oid, "label": label, "type": "portable", "is_hazard": False, "bbox": list(box), "conf": 0.8}


def test_label_from_the_list_replaces_the_label_and_marks_contacts(cfg):
    dets = [[det()], [det()]]
    contacts = [{"object_id": "obj_1", "object_label": "tool", "reasons": []},
                {"object_id": "obj_2", "object_label": "x", "reasons": []}]
    status, suggestions, changed = vlm.apply_answers({"obj_1": ("label", "angle_grinder")}, dets, contacts, VOCAB, 1e6, cfg)
    assert status == {"obj_1": "identified"} and suggestions == {} and changed == {"obj_1": "tool"}
    assert all(d[0]["label"] == "angle_grinder" and d[0]["is_hazard"] for d in dets)
    assert contacts[0]["object_label"] == "angle_grinder" and contacts[0]["reasons"] == ["label_from_vlm"]
    assert contacts[1]["reasons"] == []


def test_same_label_gets_no_reason_code_and_other_becomes_a_suggestion(cfg):
    dets = [[det(label="hammer")], [det("obj_2")]]
    contacts = [{"object_id": "obj_1", "object_label": "hammer", "reasons": []}]
    status, suggestions, changed = vlm.apply_answers({"obj_1": ("label", "hammer"), "obj_2": ("other", "chisel"),
                                                      "obj_3": ("unknown", "")}, dets, contacts, VOCAB, 1e6, cfg)
    assert contacts[0]["reasons"] == [] and changed == {}
    assert status == {"obj_1": "identified", "obj_2": "unidentified"} and suggestions == {"obj_2": "chisel"}
    assert dets[1][0]["label"] == "tool"          # the detector's label is kept


def test_a_label_that_is_not_in_the_vocabulary_is_ignored(cfg):
    dets = [[det()]]
    status, _, changed = vlm.apply_answers({"obj_1": ("label", "banana_peeler")}, dets, [], VOCAB, 1e6, cfg)
    assert status == {} and changed == {} and dets[0][0]["label"] == "tool"


def test_vlm_status_overrides_the_confidence_rule(cfg):
    frames = [{"frame_idx": i, "t_ms": i * 100, "hand_states": [],
               "objects": [{"object_id": "obj_1", "label": "tool", "type": "portable", "bbox_px": [0, 0, 5, 5],
                            "confidence": 0.9, "is_hazard": False}]} for i in range(20)]
    assert summary.build_tools(frames, [], {"tool"}, 100.0, cfg)[0]["label_status"] == "unidentified"
    assert summary.build_tools(frames, [], {"tool"}, 100.0, cfg, {"obj_1": "identified"})[0]["label_status"] == "identified"
    named = [dict(f, objects=[dict(f["objects"][0], label="hammer")]) for f in frames]
    assert summary.build_tools(named, [], {"tool"}, 100.0, cfg, {"obj_1": "unidentified"})[0]["label_status"] == "unidentified"
