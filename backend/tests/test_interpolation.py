import copy
import unittest
from backend.inference.pipeline import fill_gaps


def hand(x=200, side='left', three_d=True, size=100):
    points = [[x + (i % 5) * size / 5, 200 + (i // 5) * size / 5] for i in range(21)]
    return {'hand_id':'L' if side == 'left' else 'U1', 'side':side, 'handedness_score':0.9,
            'kp2d_px':points, 'kp3d_hand_m':[[x*.001, 0., 0.]]*21 if three_d else None,
            'bbox_px':[x, 200, x+size*.8, 200+size*.8], 'source':'detected'}


def frame(i, hands):
    return {'t_ms':i*40, 'frame_width':1920, 'frame_height':1080, 'hands':hands}


class InterpolationTests(unittest.TestCase):
    def test_short_gap_filled_without_mutating_input(self):
        raw=[frame(0,[hand(200)]),frame(1,[]),frame(2,[hand(220)])]
        original=copy.deepcopy(raw)
        result=fill_gaps(raw)
        self.assertEqual(raw,original)
        filled=result[1]['hands'][0]
        self.assertAlmostEqual(filled['kp2d_px'][0][0],210)
        self.assertAlmostEqual(filled['kp3d_hand_m'][0][0],.21)
        self.assertEqual(filled['source'],'interpolated')
        self.assertIsNone(filled['handedness_score'])

    def test_long_and_unbounded_gaps_are_not_filled(self):
        raw=[frame(0,[]),frame(1,[hand()])]+[frame(i,[]) for i in range(2,6)]+[frame(6,[hand(220)]),frame(7,[])]
        result=fill_gaps(raw)
        for i in (0,2,3,4,5,7): self.assertEqual(result[i]['hands'],[])

    def test_edge_jump_and_size_changes_are_not_filled(self):
        for a,b in ((hand(5),hand(20)),(hand(200),hand(1000)),(hand(200),hand(220,size=300))):
            self.assertEqual(fill_gaps([frame(0,[a]),frame(1,[]),frame(2,[b])])[1]['hands'],[])

    def test_overlapping_unknown_hand_is_not_duplicated(self):
        unknown=hand(210,side='unknown')
        raw=[frame(0,[hand(200)]),frame(1,[unknown]),frame(2,[hand(220)])]
        self.assertEqual(fill_gaps(raw)[1]['hands'],[unknown])

    def test_measured_2d_remains_measured_when_only_3d_is_filled(self):
        measured=hand(210,three_d=False)
        raw=[frame(0,[hand(200)]),frame(1,[measured]),frame(2,[hand(220)])]
        result=fill_gaps(raw)[1]['hands'][0]
        self.assertEqual(result['kp2d_px'],measured['kp2d_px'])
        self.assertEqual(result['source'],'detected')
        self.assertEqual(result['kp3d_hand_m_source'],'interpolated')
        self.assertAlmostEqual(result['kp3d_hand_m'][0][0],.21)

    def test_duplicate_timestamps_are_not_filled(self):
        middle=frame(1,[]); middle['t_ms']=0
        self.assertEqual(fill_gaps([frame(0,[hand()]),middle,frame(2,[hand(220)])])[1]['hands'],[])
