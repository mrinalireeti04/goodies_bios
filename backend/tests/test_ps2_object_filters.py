import unittest
import numpy as np

from ps2.ledger import ledger_vocab
from ps2.objects import filter_detections


class ObjectFilterTests(unittest.TestCase):
    def test_workpieces_are_available_to_detector(self):
        ledger = {'running': [], 'tools': [{'name': 'aluminum profile', 'synonyms': ['metal channel'],
                  'type': 'portable', 'workpiece': True, 'powered': False, 'tool_key': 'profile_1'}]}
        vocab = ledger_vocab(ledger, {'objects': {}, 'generic': {}, 'ignore': []})
        self.assertEqual(vocab['objects']['aluminum_profile']['prompts'], ['aluminum profile', 'metal channel'])
        self.assertTrue(vocab['objects']['aluminum_profile']['workpiece'])

    def test_filter_rejects_weak_oversized_and_invalid_boxes(self):
        boxes = np.array([[0, 0, 20, 20], [0, 0, 20, 20], [0, 0, 100, 100],
                          [10, 10, 10, 20], [-10, -10, 30, 30]], dtype=float)
        scores = np.array([.5, .2, .9, .9, .4])
        ids = np.arange(len(boxes))
        filtered, _, kept = filter_detections(boxes, scores, ids, 100, 100, .25, .75)
        self.assertEqual(kept.tolist(), [0, 4])
        self.assertEqual(filtered[1].tolist(), [0, 0, 30, 30])


if __name__ == '__main__':
    unittest.main()
