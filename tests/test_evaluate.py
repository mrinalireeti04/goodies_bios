import csv
import json
import sys
import urllib.error
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dev"))

import baseline_v0  # noqa: E402
import checklist as cl  # noqa: E402
import evaluate as ev  # noqa: E402
import groq_judge as gj  # noqa: E402
import independent_proposals as ip  # noqa: E402
import verify_contacts as vc  # noqa: E402
from ps2.hands import Hand  # noqa: E402
from test_hands import make_kp  # noqa: E402


# ---------------------------------------------------------------- names, hands, pooling

def test_names_match_by_equality_containment_plural_and_name_map():
    assert cl.names_match("Claw Hammer", "hammer") and cl.names_match("power_drill", "power drill") and cl.names_match("pliers", "plier")
    assert not cl.names_match("hammer", "saw") and not cl.names_match("hammer", "unknown") and not cl.names_match("", "hammer")
    groups = [{cl.canonical("mallet"), cl.canonical("rubber hammer")}]
    assert cl.names_match("mallet", "rubber hammer", groups) and not cl.names_match("mallet", "rubber hammer", [])


def sysc(start, end, hand="h_left", label="hammer", obj="obj_1", inter="hold"):
    return {"start_ms": start, "end_ms": end, "hand_id": hand, "object_label": label, "object_id": obj, "interaction": inter,
            "state": "portable", "event_id": f"c_{start}"}


def prop(start, end, hand="left", obj="hammer"):
    return {"start_ms": start, "end_ms": end, "hand": hand, "object": obj, "action": "hit", "certainty": "high"}


def test_pooling_matches_by_overlap_and_hand_and_keeps_unmatched_from_each_side():
    items = cl.pool([sysc(1000, 3000), sysc(8000, 9000, hand="h_right"), sysc(20000, 21000)],
                    [prop(1200, 3200), prop(8100, 9100, hand="left"), prop(30000, 31000)], 0.3)
    assert [(i["source"], i["start_ms"]) for i in items] == [("both", 1000), ("system", 8000), ("gemini_pro", 8100), ("system", 20000), ("gemini_pro", 30000)]
    assert items[2]["hand"] == "left" and items[1]["hand"] == "right"       # the right-hand system contact and the left-hand proposal do not merge
    one_to_one = cl.pool([sysc(1000, 3000)], [prop(1000, 3000), prop(1100, 3100)], 0.3)
    assert sorted(i["source"] for i in one_to_one) == ["both", "gemini_pro"]


def test_tiers_votes_and_names():
    groups = None

    def tier(system, gemini, groq, names=None):
        v = {"system": system, "gemini": gemini, "groq": groq, "names": names or {}}
        return cl.tier_of(v, groups)

    assert tier("yes", "no", "unsure") == 1                                               # three different answers
    assert tier("yes", "yes", "yes", {"system": "hammer", "gemini": "claw hammer", "groq": "hammer"}) == 3
    assert tier("yes", "yes", "yes", {"system": "hammer", "gemini": "saw", "groq": "hammer"}) == 2   # contact agreed, name not
    assert tier("yes", "yes", "yes", {"system": "hammer", "gemini": "saw", "groq": "drill"}) == 1    # three different names
    assert tier("yes", "no", "no") == 2 and tier("no", "yes", "yes") == 2 and tier("yes", "yes", "no") == 2
    assert tier("yes", "no", None) == 2 and tier("yes", "yes", None, {"system": "hammer", "gemini": "hammer"}) == 3


def fake_items(n1, n2, n3):
    items = [{"key": f"k{i}", "start_ms": i * 1000, "end_ms": i * 1000 + 500} for i in range(n1 + n2 + n3)]
    tiers = [1] * n1 + [2] * n2 + [3] * n3
    return items, tiers


def test_selection_orders_tiers_reserves_agreements_and_samples_with_a_fixed_seed():
    items, tiers = fake_items(3, 10, 50)
    chosen, strata = cl.select(items, tiers, 20, seed=7)
    assert len(chosen) == 20 and strata == {"1": {"N": 3, "n": 3}, "2": {"N": 10, "n": 9}, "3": {"N": 50, "n": 8}}
    order = [tiers[items.index(c)] for c in chosen]
    assert order == sorted(order)                                       # tier 1, then 2, then 3
    again, _ = cl.select(items, tiers, 20, seed=7)
    other, _ = cl.select(items, tiers, 20, seed=8)
    assert again == chosen and other != chosen                          # reproducible, and the seed matters
    few, strata = cl.select(*fake_items(1, 2, 3), 40, seed=7)
    assert len(few) == 6 and strata["3"] == {"N": 3, "n": 3}


def test_checklist_keeps_the_population_and_the_judges_answers(cfg):
    groq = {"system|1000|3000": {"contact": "yes", "hand": "left", "tool": "hammer"}}
    out = cl.build_checklist("clip", "ledger_sam", [sysc(1000, 3000)], [prop(1000, 3000), prop(9000, 9500, obj="saw")], groq, cfg,
                             status_of={"obj_1": {"identified": True, "flagged": False}})
    assert out["n_pooled"] == 2 and len(out["items"]) == 2 and out["seed"] == cfg["checklist_seed"]
    both = next(i for i in out["items"] if i["source"] == "both")
    assert both["system"]["object_id"] == "obj_1" and both["gemini"]["object"] == "hammer"
    pop = {p["key"]: p for p in out["population"]}
    assert pop[both["key"]]["identified"] is True and pop[both["key"]]["votes"]["groq"] is None   # key differs: this one was not judged


def test_the_verify_window_gets_the_source_and_its_own_guess_but_not_the_other_models(cfg):
    out = cl.build_checklist("clip", "v", [sysc(1000, 3000)], [prop(1000, 3000, obj="claw hammer"), prop(9000, 9500, obj="saw")],
                             {}, cfg)
    shown = vc.checklist_items(out)
    assert {s["source"] for s in shown} == {"both", "gemini_pro"}
    both = next(s for s in shown if s["source"] == "both")
    assert both["object_label"] == "hammer"                              # the system's own name, not Gemini's "claw hammer"
    text = json.dumps(shown)
    assert "groq" not in text and "votes" not in text and "claw hammer" not in json.dumps(both)


def test_check_answers_are_saved_with_source_and_tier_and_resume(tmp_path, cfg):
    out = cl.build_checklist("clip", "v", [sysc(1000, 3000)], [prop(8000, 9000, obj="saw")], {}, cfg)
    items = vc.checklist_items(out)
    s = vc.VerifySession("check", items, tmp_path / "verify_clip.csv")
    s.record(items[0], "y")
    s.record(items[1], "o", "chisel")
    again = vc.VerifySession("check", items, tmp_path / "verify_clip.csv")
    assert again.pending() == []
    with (tmp_path / "verify_clip.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["source"], r["verdict"]) for r in rows] == [("system", "y"), ("gemini_pro", "o")] and rows[1]["correct_name"] == "chisel"
    with pytest.raises(ValueError):
        s.record(items[0], "z")


# ---------------------------------------------------------------- stratified estimates

def checklist_with(population, items):
    return {"population": population, "items": items, "strata": {}, "n_pooled": len(population), "seed": 7}


def pop_entry(key, tier, source, identified=True, flagged=False):
    return {"key": key, "tier": tier, "source": source, "identified": identified, "flagged": flagged,
            "votes": {"system": "yes", "gemini": "yes", "groq": "yes", "names": {}}}


def test_precision_is_weighted_by_tier_size_not_by_how_many_were_listed():
    # 90 unanimous system contacts (tier 3), 10 disputed ones (tier 1). We listed all 10 disputed and only 10 of the 90.
    population = [pop_entry(f"a{i}", 3, "both") for i in range(90)] + [pop_entry(f"b{i}", 1, "system") for i in range(10)]
    items = [{"key": f"a{i}", "tier": 3, "source": "both"} for i in range(10)] + [{"key": f"b{i}", "tier": 1, "source": "system"} for i in range(10)]
    answers = {f"a{i}": {"verdict": "y"} for i in range(10)} | {f"b{i}": {"verdict": "n"} for i in range(10)}
    out = ev.human_metrics(checklist_with(population, items), answers, seed=7)
    assert out["precision"]["value"] == pytest.approx(0.9)               # a naive "10 of 20" would say 0.5
    assert out["precision"]["low"] <= 0.9 <= out["precision"]["high"] and out["precision"]["n"] == 20
    assert out["counts"] == {"y": 10, "o": 0, "i": 0, "n": 10}


def test_naming_identified_flagged_and_recall_estimates():
    population = ([pop_entry(f"s{i}", 2, "system") for i in range(10)] + [pop_entry(f"g{i}", 2, "gemini_pro", identified=False) for i in range(10)])
    items = [{"key": k["key"], "tier": 2, "source": k["source"]} for k in population]
    verdicts = {}
    for i in range(10):                    # system items: 6 correct, 2 wrong object (one flagged), 2 not real
        verdicts[f"s{i}"] = "y" if i < 6 else "o" if i < 8 else "n"
    for i in range(10):                    # Gemini-only items: 5 real (the system missed them), 5 not real
        verdicts[f"g{i}"] = "y" if i < 5 else "n"
    population[6]["flagged"] = True
    population[7]["identified"] = False
    answers = {k: {"verdict": v} for k, v in verdicts.items()}
    out = ev.human_metrics(checklist_with(population, items), answers, seed=7)
    assert out["precision"]["value"] == pytest.approx(0.8)                     # 8 of 10 system items are real
    assert out["naming_accuracy"]["value"] == pytest.approx(6 / 8)
    assert out["identified_precision"]["value"] == pytest.approx(6 / 7)        # one wrong-object item was marked unidentified
    assert out["wrong_names_flagged"]["value"] == pytest.approx(1.0)           # both wrong names were flagged or unidentified
    assert out["recall_of_pooled_real"]["value"] == pytest.approx(8 / 13)      # 8 real found by the system of 13 real in the pool
    assert out["gemini_pro_precision"]["value"] == pytest.approx(0.5)


def test_tiers_without_answers_are_reported_and_no_answers_gives_nothing_to_estimate():
    population = [pop_entry("a", 3, "both"), pop_entry("b", 1, "system")]
    items = [{"key": "a", "tier": 3, "source": "both"}, {"key": "b", "tier": 1, "source": "system"}]
    out = ev.human_metrics({**checklist_with(population, items), "strata": {"1": {"N": 1, "n": 1}, "3": {"N": 1, "n": 1}}}, {"a": {"verdict": "y"}}, 7)
    assert out["missing_tiers"] == ["1"] and out["precision"]["value"] == 1.0
    assert ev.human_metrics(checklist_with(population, items), {}, 7)["answered"] == 0


def test_agreement_is_counted_over_all_pooled_items_and_skips_unsure():
    def entry(system, gemini, groq, names=None):
        return {"key": "k", "tier": 2, "source": "both", "votes": {"system": system, "gemini": gemini, "groq": groq, "names": names or {}}}

    population = [entry("yes", "yes", "yes", {"system": "hammer", "gemini": "claw hammer", "groq": "saw"}), entry("yes", "no", "no"),
                  entry("no", "yes", "unsure"), entry("yes", "yes", None)]
    a = ev.agreement({"population": population}, None)
    assert a["system_vs_gemini_pro"]["agree"] == 2 and a["system_vs_gemini_pro"]["of"] == 4
    assert a["system_vs_groq"] == {"agree": 1, "of": 2, "skipped_unsure_or_unjudged": 2, "same_tool_name_when_both_say_contact": "0/1"}
    assert a["system_vs_gemini_pro"]["same_tool_name_when_both_say_contact"] == "1/1"          # hammer ~ claw hammer
    assert "AGREEMENT" in a["source"]


# ---------------------------------------------------------------- exact recall and timing in a labelled window

def label(start, end, hand="left", obj="hammer", inter="hold", unsure="0", state="portable"):
    return {"kind": "contact", "start_ms": start, "end_ms": end, "hand": hand, "state": state, "interaction": inter,
            "object": obj, "instance": "", "unsure": unsure}


def test_window_metrics_exclude_unsure_labels_and_spans_and_measure_timing():
    labels = [label(1000, 3000), label(5000, 6000), label(7000, 8000, unsure="1"), label(12000, 13000)]
    spans = [{"kind": "unsure_span", "start_ms": 11500, "end_ms": 14000}]
    system = [sysc(1100, 3200), sysc(7000, 8000), sysc(12000, 13000), sysc(20000, 21000)]   # only the first two windows' contacts count
    out = ev.window_metrics(labels, spans, system, (0, 15000), None)
    assert out["labelled"] == 2 and out["predicted"] == 2          # the unsure label, the label inside the span and the span's prediction are gone
    assert out["iou0.5_hand"]["recall"] == "1/2" and out["iou0.5_hand"]["precision"] == "1/2"
    t = out["timing_ms_at_iou0.3"]
    assert t["matched"] == 1 and t["start_median"] == 100 and t["end_median"] == 200 and out["name_correct"] == "1/1"
    assert ev.window_metrics([], [], system, None, None) is None


def test_report_says_where_every_number_comes_from(cfg):
    population = [pop_entry("a", 3, "both"), pop_entry("b", 1, "system")]
    items = [{"key": "a", "tier": 3, "source": "both", "id": "k1"}, {"key": "b", "tier": 1, "source": "system", "id": "k2"}]
    checklist = {**checklist_with(population, items), "strata": {"1": {"N": 1, "n": 1}, "3": {"N": 1, "n": 1}}}
    human = ev.human_metrics(checklist, {"a": {"verdict": "y"}, "b": {"verdict": "o"}}, 7)
    text = ev.report("clip", "v", human, ev.agreement(checklist, None), None, checklist)
    assert "misses by every model are not counted" in text and "AGREEMENT, not accuracy" in text and "95% CI" in text
    assert "your answers on the check list" in text


# ---------------------------------------------------------------- Groq judge

def test_groq_answers_are_read_even_with_thinking_text_and_fences():
    ok = gj.parse_answer('<think>hmm</think>```json\n{"contact": "Yes", "hand": "LEFT", "tool": "  Claw  Hammer "}\n```')
    assert ok == {"contact": "yes", "hand": "left", "tool": "claw hammer"}
    assert gj.parse_answer('{"contact": "maybe"}') is None and gj.parse_answer("no json") is None and gj.parse_answer(None) is None
    assert gj.parse_answer('{"contact": "no", "hand": "sideways"}')["hand"] == "unsure"


def test_rate_limit_durations():
    assert gj.parse_duration("2.5s") == 2.5 and gj.parse_duration("250ms") == 0.25 and gj.parse_duration("1m3.5s") == 63.5
    assert gj.parse_duration(None) is None and gj.parse_duration("soon") is None


def test_plan_steps_down_from_three_frames_to_one_to_disagreements_only():
    assert gj.plan(40, 10, 900, 450_000, 900) == ("3", 40)
    assert gj.plan(40, 10, 900, 40 * 900 * 2, 900)[0] == "1"                       # not enough tokens for 3 frames each
    mode, count = gj.plan(40, 10, 900, 6 * (900 + gj.OVERHEAD_TOKENS), 900)
    assert mode == "disagree" and count == 6
    assert gj.plan(40, 10, 5, 450_000, 900) == ("disagree", 5)                       # few requests left
    assert gj.plan(40, 10, 0, 0, 900) == ("disagree", 0)


class FakeGroq:
    def __init__(self, *items):
        self.items, self.calls = list(items), []

    def __call__(self, url, headers, body, timeout):
        self.calls.append((url, headers, json.loads(body)))
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def groq_reply(contact="yes", tokens=1500, headers=None):
    content = json.dumps({"contact": contact, "hand": "left", "tool": "hammer"})
    return headers or {}, json.dumps({"choices": [{"message": {"content": content}}], "usage": {"total_tokens": tokens}}).encode()


def http_429(retry_after):
    return urllib.error.HTTPError("u", 429, "slow down", {"retry-after": str(retry_after)}, None)


def judge(tmp_path, cfg, fake, key="k"):
    waits = []
    j = gj.GroqJudge(cfg, tmp_path, key=key, http=fake, sleep=waits.append)
    j.waits = waits
    return j


def test_a_request_has_at_most_three_images_the_model_and_temperature_zero(tmp_path, cfg):
    fake = FakeGroq(groq_reply())
    j = judge(tmp_path, cfg, fake)
    assert j.ask(["a", "b", "c", "d", "e"], "the middle", ["hammer", "saw"])["contact"] == "yes"
    url, headers, body = fake.calls[0]
    content = body["messages"][0]["content"]
    assert sum(c["type"] == "image_url" for c in content) == 3 and body["model"] == cfg["groq_model"] and body["temperature"] == 0
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,") and "hammer, saw" in content[0]["text"]
    assert headers["Authorization"] == "Bearer k" and j.usage.requests == 1 and j.usage.tokens == 1500


def test_short_429_is_waited_out_and_a_long_one_means_the_daily_limit(tmp_path, cfg):
    j = judge(tmp_path, cfg, FakeGroq(http_429(3), groq_reply()))
    assert j.ask(["a"], "x", []) is not None and j.waits == [3]
    j = judge(tmp_path, cfg, FakeGroq(http_429(3600), groq_reply()))
    assert j.ask(["a"], "x", []) is None and j.exhausted and not j.available
    assert j.ask(["a"], "x", []) is None


def test_a_token_window_wait_is_slept_and_then_judging_uses_one_frame(tmp_path, cfg):
    j = judge(tmp_path, cfg, FakeGroq(http_429(724), groq_reply()))
    assert j.ask(["a"], "x", []) is not None
    assert j.waits == [724] and j.stepped_down and j.available
    asks = [{"key": f"k{i}", "start_ms": 0, "end_ms": 1000, "disagree": False, "names": []} for i in range(3)]
    fake = FakeGroq(groq_reply(), http_429(724), groq_reply(), groq_reply())
    j2 = judge(tmp_path / "b", cfg, fake)
    seen = []
    answers = gj.judge_items(j2, asks, lambda it, n: seen.append(n) or ["img"] * n, tmp_path / "c2.json")
    assert seen == [3, 3, 1] and {k: v["frames"] for k, v in answers.items()} == {"k0": 3, "k1": 3, "k2": 1}


def test_unknown_daily_token_limit_does_not_block_planning(tmp_path, cfg):
    j = judge(tmp_path, {**cfg, "groq_max_tokens_per_day": 0}, FakeGroq())
    j.usage.add(10 ** 6)
    assert j.budget()[1] > 10 ** 8


def test_bad_key_stops_and_no_key_never_calls(tmp_path, cfg):
    fake = FakeGroq(urllib.error.HTTPError("u", 401, "no", {}, None))
    j = judge(tmp_path, cfg, fake)
    assert j.ask(["a"], "x", []) is None and j.exhausted
    quiet = judge(tmp_path, cfg, FakeGroq(), key="")
    assert quiet.ask(["a"], "x", []) is None


def test_low_remaining_tokens_in_the_headers_make_it_wait_for_the_reset(tmp_path, cfg):
    j = judge(tmp_path, cfg, FakeGroq(groq_reply(headers={"x-ratelimit-remaining-tokens": "100", "x-ratelimit-reset-tokens": "4.5s"})))
    j.ask(["a"], "x", [])
    assert j.waits == [4.5]


def test_usage_is_kept_per_day_and_judging_logs_the_mode_and_caches_only_json(tmp_path, cfg):
    asks = [{"key": f"k{i}", "start_ms": 0, "end_ms": 1000, "disagree": i < 2, "names": ["hammer"]} for i in range(4)]
    fake = FakeGroq(*[groq_reply() for _ in range(4)])
    j = judge(tmp_path, cfg, fake)
    cache = tmp_path / "groq_clip.json"
    answers = gj.judge_items(j, asks, lambda it, n: ["img"] * n, cache)
    assert set(answers) == {"k0", "k1", "k2", "k3"} and json.loads(cache.read_text(encoding="utf-8"))["mode"] == "3"
    assert len(fake.calls[0][2]["messages"][0]["content"]) == 1 + 3
    again = gj.judge_items(judge(tmp_path, cfg, FakeGroq()), asks, lambda it, n: ["img"] * n, cache)   # nothing left to ask
    assert again == answers
    assert gj.Usage(tmp_path / "groq_usage.json").requests == 4
    assert gj.Usage(tmp_path / "groq_usage.json", today="2000-01-01").requests == 0                     # a new day starts at zero


def test_a_small_budget_falls_back_to_disagreements_with_one_frame(tmp_path, cfg):
    asks = [{"key": f"k{i}", "start_ms": 0, "end_ms": 1000, "disagree": i in (1, 3), "names": []} for i in range(4)]
    small = {**cfg, "groq_max_requests_per_day": 2}
    fake = FakeGroq(groq_reply(), groq_reply())
    answers = gj.judge_items(judge(tmp_path, small, fake), asks, lambda it, n: ["img"] * n, tmp_path / "c.json")
    assert set(answers) == {"k1", "k3"} and all(len(c[2]["messages"][0]["content"]) == 2 for c in fake.calls)


# ---------------------------------------------------------------- independent proposals

def test_pro_models_are_tried_stable_first_and_aliases_last():
    names = ["gemini-3.6-flash", "gemini-3.1-pro-preview", "gemini-2.5-pro", "gemini-pro-latest", "gemini-3-pro-image",
             "gemini-2.5-pro-preview-tts", "gemini-3.1-pro-preview-customtools", "gemini-3.5-flash-lite"]
    assert ip.pro_candidates(names) == ["gemini-2.5-pro", "gemini-3.1-pro-preview", "gemini-pro-latest"]


def test_proposals_are_validated_and_sorted():
    data = {"contacts": [{"start": "00:20", "end": "00:25", "hand": "Left", "tool": " Claw  Hammer ", "action": "Hit", "certainty": "high"},
                         {"start": "00:05", "end": "00:08", "hand": "right", "tool": "saw", "action": "cut", "certainty": "weird"},
                         {"start": "00:10", "end": "00:09", "hand": "left", "tool": "x", "action": "", "certainty": "low"},
                         {"start": "09:00", "end": "09:05", "hand": "left", "tool": "x", "action": "", "certainty": "low"},
                         {"start": "00:30", "end": "00:31", "hand": "left", "tool": "", "action": "", "certainty": "low"}, "junk"]}
    out = ip.parse_proposals(data, 60.0)
    assert [(c["start_ms"], c["end_ms"], c["hand"], c["object"], c["certainty"]) for c in out] == \
           [(5000, 8000, "right", "saw", "low"), (20000, 25000, "left", "claw hammer", "high")]
    assert ip.parse_proposals({"nothing": 1}, 60.0) == [] and ip.parse_proposals(None, 60.0) == []


class FakeFiles:
    def __init__(self):
        self.deleted = []

    def upload(self, path, mime, name):
        return {"name": "files/x", "uri": "https://files/x"}

    def wait_active(self, name):
        return {"state": "ACTIVE", "uri": "https://files/x"}

    def delete(self, name):
        self.deleted.append(name)
        return True


class FakeProGem:
    """Models in `unavailable` answer nothing; the others answer with one proposal."""
    unavailable = {"gemini-2.5-pro", "gemini-3.1-pro-preview"}
    log: list = []

    def __init__(self, cfg):
        self.model, self.calls, self.available, self.key = cfg["vlm_model"], 0, True, "k"
        FakeProGem.log.append((cfg["vlm_model"], cfg["vlm_fallback_model"], cfg["vlm_thinking_level"]))

    def generate(self, parts, max_tokens, schema=None, timeout=None):
        self.calls += 1
        if self.model in self.unavailable:
            return None
        return json.dumps({"contacts": [{"start": "00:10", "end": "00:12", "hand": "left", "tool": "hammer", "action": "hit", "certainty": "high"}]})


def propose(cfg, models, files):
    return ip.propose(Path("clip.mp4"), 100.0, {**cfg, "independent_max_calls": 6}, models, Path("unused"),
                      files=files, make_copy=None, make_gemini=FakeProGem)


def test_independent_proposals_upload_once_fall_back_from_pro_to_another_flash_and_delete_the_upload(cfg):
    FakeProGem.log = []
    files = FakeFiles()
    out = propose(cfg, ["gemini-2.5-pro", "gemini-3.1-pro-preview", "gemini-3.8-flash", "gemini-3.6-flash"], files)
    assert out["model"] == "gemini-3.8-flash" and out["run"]["kind"] == "flash" and len(out["contacts"]) == 1
    assert out["run"]["tried"] == ["gemini-2.5-pro", "gemini-3.1-pro-preview", "gemini-3.8-flash"]    # never the pipeline's own gemini-3.6-flash
    assert set(out["run"]["failed"]) == {"gemini-2.5-pro", "gemini-3.1-pro-preview"}
    assert files.deleted == ["files/x"] and out["run"]["upload_deleted"] is True                     # uploaded once, deleted once
    pro = next(entry for entry in FakeProGem.log if entry[0] == "gemini-2.5-pro")
    assert pro[1:] == (None, None)                                   # no fallback model; no thinking override for Pro
    flash = next(entry for entry in FakeProGem.log if entry[0] == "gemini-3.8-flash")
    assert flash[2] == cfg["vlm_thinking_level"]


def test_a_working_pro_model_is_preferred_and_no_working_model_still_cleans_up(cfg):
    FakeProGem.unavailable = {"gemini-2.5-pro"}
    out = propose(cfg, ["gemini-2.5-pro", "gemini-3.1-pro-preview", "gemini-3.8-flash"], FakeFiles())
    assert out["model"] == "gemini-3.1-pro-preview" and out["run"]["kind"] == "pro"
    FakeProGem.unavailable = {"gemini-2.5-pro", "gemini-3.1-pro-preview"}
    files = FakeFiles()
    out = propose(cfg, ["gemini-2.5-pro"], files)
    assert out["model"] is None and out["contacts"] == [] and files.deleted == ["files/x"]


# ---------------------------------------------------------------- the old contact rule, for comparison

def test_v0_rule_baseline_makes_intervals_from_a_finished_result(cfg):
    kp = make_kp()
    hands = {i: [Hand("h_right", kp, np.array([30, -60, 150, 110.0]), "detected")] for i in range(5, 30)}
    box = [30, -70, 150, 70]  # all five fingertips inside
    frames = [{"frame_idx": i, "t_ms": round(i * 100), "hand_states": [],
               "objects": [{"object_id": "obj_1", "label": "hammer", "type": "portable", "bbox_px": box,
                            "confidence": 0.8}]} for i in range(40)]
    out = baseline_v0.v0_contacts({"frames": frames}, hands, cfg)
    assert len(out) == 1 and out[0]["start_ms"] == 500 and out[0]["end_ms"] == 3000
    assert out[0]["object_id"] == "obj_1" and out[0]["interaction"] == "unknown"
