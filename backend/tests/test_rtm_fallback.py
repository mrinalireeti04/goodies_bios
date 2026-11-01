import math

from backend.inference.pipeline import needs_rtm_fallback


def test_confident_mediapipe_hands_skip_rtm():
    assert not needs_rtm_fallback([{'side': 'left', 'handedness_score': 0.91}])


def test_missing_or_uncertain_hands_use_rtm():
    assert needs_rtm_fallback([])
    for hand in (
        {'side': 'left', 'handedness_score': 0.79},
        {'side': 'unknown', 'handedness_score': 0.95},
        {'side': 'right', 'handedness_score': None},
        {'side': 'right', 'handedness_score': math.nan},
    ):
        assert needs_rtm_fallback([hand])


def test_one_uncertain_hand_triggers_frame_fallback():
    assert needs_rtm_fallback([
        {'side': 'left', 'handedness_score': 0.95},
        {'side': 'right', 'handedness_score': 0.6},
    ])
