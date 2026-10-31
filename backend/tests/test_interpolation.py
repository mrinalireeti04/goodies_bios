import unittest
from backend.inference.pipeline import fill_gaps

def frame(i, present):
    return {'t_ms':i*40,'hands':[{'hand_id':'L','side':'left','kp2d_px':[[float(i),float(i)]]*21,'kp3d_hand_m':[[i*.01,0.,0.]]*21,'bbox_px':[i,i,i,i],'source':'detected'}] if present else []}

class InterpolationTests(unittest.TestCase):
    def test_ten_missing_frames_are_filled_without_mutating_raw(self):
        raw=[frame(i,i in (0,11)) for i in range(12)]
        result=fill_gaps(raw)
        self.assertEqual(raw[5]['hands'],[])
        self.assertAlmostEqual(result[5]['hands'][0]['kp2d_px'][0][0],5)
        self.assertAlmostEqual(result[5]['hands'][0]['kp3d_hand_m'][0][0],.05)
        self.assertEqual(result[5]['hands'][0]['source'],'interpolated')

    def test_eleven_missing_frames_are_not_filled(self):
        result=fill_gaps([frame(i,i in (0,12)) for i in range(13)])
        self.assertEqual(result[5]['hands'],[])

    def test_unbounded_gaps_remain_empty(self):
        result=fill_gaps([frame(i,i==3) for i in range(8)])
        self.assertEqual(result[0]['hands'],[])
        self.assertEqual(result[-1]['hands'],[])
