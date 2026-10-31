import json
import tempfile
import unittest
from pathlib import Path
from modules.ps4 import run_ps4

class PS4ContractTest(unittest.TestCase):
    def test_pose_export_and_missing_hand(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)
            input_path=directory/'hands.jsonl'
            hand={'hand_id':'H0','side':'right','kp3d_hand_m':[[0.001*i,0.002*i,0.003*i] for i in range(21)],'source':'detected'}
            records=[{'clip_id':'synthetic-test','frame_idx':i,'t_ms':i*40,'hands':[hand] if i!=1 else []} for i in range(3)]
            input_path.write_text('\n'.join(json.dumps(r) for r in records),encoding='utf-8')
            dest=run_ps4(input_path,directory/'results')
            data=json.loads(dest.read_text(encoding='utf-8'))
            self.assertEqual(data['coordinate_frame'],'hand_centered')
            self.assertEqual(len(data['frames']),3)
            self.assertEqual(len(data['frames'][0]['hands'][0]['joints_3d_m']),21)
            self.assertEqual(data['frames'][1]['hands'],[])
            self.assertEqual(data['frames'][2]['hands'][0]['hand_id'],'H0')
            self.assertEqual(data['render_fps'],25.)

if __name__=='__main__':unittest.main()
