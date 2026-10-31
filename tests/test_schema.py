import copy

import pytest

from ps2 import schema


def good():
    return schema.build_output(
        "clip",
        [{"frame_idx": 0, "t_ms": 0,
          "objects": [{"object_id": "obj_1", "label": "hammer", "type": "portable",
                       "bbox_px": [0, 0, 10, 10], "confidence": 0.5}],
          "hand_states": [{"hand_id": "h_left", "state": "portable", "object_id": "obj_1", "confidence": 0.7},
                          {"hand_id": "h_right", "state": "none", "object_id": None, "confidence": None}]}],
        [{"event_id": "c_1", "start_ms": 0, "end_ms": 100, "hand_id": "h_left", "object_id": "obj_1",
          "object_label": "hammer", "state": "portable", "interaction": "unknown", "confidence": 0.7,
          "_first": 0}],
    )


def test_good_output_passes_and_private_keys_are_stripped():
    out = good()
    schema.validate(out)
    assert "_first" not in out["contacts"][0]


def bad(mutate):
    out = copy.deepcopy(good())
    mutate(out)
    with pytest.raises(ValueError):
        schema.validate(out)


def test_enums():
    bad(lambda o: o["frames"][0]["objects"][0].update(type="heavy"))
    bad(lambda o: o["frames"][0]["hand_states"][0].update(state="holding"))
    bad(lambda o: o["contacts"][0].update(state="unknown"))
    bad(lambda o: o["contacts"][0].update(interaction="grab"))


def test_none_state_needs_null_object():
    bad(lambda o: o["frames"][0]["hand_states"][1].update(object_id="obj_1"))


def test_interval_order():
    bad(lambda o: o["contacts"][0].update(start_ms=100, end_ms=100))


def test_referenced_objects_must_exist():
    bad(lambda o: o["contacts"][0].update(object_id="obj_9"))
    bad(lambda o: o["frames"][0]["hand_states"][0].update(object_id="obj_9"))


def test_required_fields():
    bad(lambda o: o.pop("contacts"))
    bad(lambda o: o["frames"][0]["objects"][0].pop("bbox_px"))
    bad(lambda o: o["contacts"][0].pop("event_id"))


def test_optional_fields_can_be_absent():
    out = good()
    assert "meta" not in out and "tools" not in out
    schema.validate(out)
