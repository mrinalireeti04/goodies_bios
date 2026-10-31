"""Local API smoke tests; no private footage and no teammate dependencies required."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
import backend.app as backend


class LocalAPITest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.client=TestClient(backend.app)
        self._data=patch.object(backend,'DATA_DIR',Path(self.temp.name))
        self._data.start()
        backend.JOBS.clear()

    def tearDown(self):
        backend.JOBS.clear()
        self._data.stop()
        self.temp.cleanup()

    def test_health_and_invalid_file(self):
        self.assertTrue(self.client.get('/api/health').json()['ok'])
        r=self.client.post('/api/jobs',files={'file':('document.exe',b'junk','application/octet-stream')})
        self.assertEqual(r.status_code,415)

    def test_local_media_upload_no_fabricated_predictions_and_delete(self):
        res=self.client.post('/api/jobs',files={'file':('some_photo.png',b'\x89PNG\r\n\x1a\n','image/png')},data={'client_clip_id':'CLIP1'})
        self.assertEqual(res.status_code,201,res.text)
        job=res.json()
        self.assertEqual(job['client_clip_id'],'CLIP1')
        job_id=job['job_id']
        state=self.client.get('/api/jobs/'+job_id).json()
        self.assertEqual(state['status'],'partial')
        self.assertEqual(state['completed_modules'],[])
        self.assertEqual(self.client.get('/api/clips/'+job_id+'/hands').status_code,404)
        self.assertEqual(self.client.get('/api/clips/'+job_id+'/media').status_code,200)
        self.assertTrue((Path(self.temp.name)/job_id).exists())
        self.assertEqual(self.client.delete('/api/jobs/'+job_id).status_code,204)
        self.assertFalse((Path(self.temp.name)/job_id).exists())

    def test_real_ps4_runs_after_ps1_when_other_modules_missing(self):
        import sys
        from types import ModuleType
        fake=ModuleType('modules.ps1')
        def run_ps1(video_path, output_dir):
            path=Path(output_dir)/'hands.jsonl'
            rows=[]
            for i in range(3):
                rows.append({'schema_version':'1.0','clip_id':'sample','frame_idx':i,
                             't_ms':i*40,'frame_width':480,'frame_height':270,
                             'hands':[{'hand_id':'H0','side':'left','source':'detected',
                             'kp3d_hand_m':[[0.001*k,0.002*k,0.003*k] for k in range(21)]} ]})
            path.write_text('\n'.join(json.dumps(r) for r in rows))
            return path
        fake.run_ps1=run_ps1
        with patch.dict(sys.modules,{'modules.ps1':fake}):
            response=self.client.post('/api/jobs',files={'file':('factory.mp4',b'fake data','video/mp4')})
            jid=response.json()['job_id']
            job=self.client.get(f'/api/jobs/{jid}').json()
            self.assertEqual(job['status'],'partial')
            self.assertEqual(job['completed_modules'],['ps1','ps4'])
            poses=self.client.get(f'/api/clips/{jid}/pose3d').json()
            self.assertEqual(len(poses['frames']),3)
            self.assertEqual(len(poses['frames'][0]['hands'][0]['joints_3d_m']),21)
            self.assertEqual(self.client.delete(f'/api/jobs/{jid}').status_code,204)

    def test_video_job_and_timestamp_scoped_results(self):
        res=self.client.post('/api/jobs',files={'file':('factory.mp4',b'not real footage','video/mp4')})
        jid=res.json()['job_id']
        state=self.client.get('/api/jobs/'+jid).json()
        self.assertEqual(state['status'],'partial')
        self.assertEqual(state['completed_modules'],[])
        results=Path(self.temp.name)/jid/'results'
        results.mkdir(exist_ok=True)
        p=results/'hands.jsonl'
        p.write_text('\n'.join(json.dumps({'clip_id':'factory','frame_idx':i,'t_ms':i*1000,'hands':[]}) for i in range(3)))
        (results/'actions.json').write_text(json.dumps({'clip_id':'factory','actions':[{'action_id':'A1','start_ms':500,'end_ms':1500,'verb':'fasten','object_label':'screwdriver','hand':'right','hand_ids':[],'confidence':None}]}))
        backend.put(jid,completed_modules=['ps1','ps3'])
        self.assertEqual(len(self.client.get(f'/api/clips/{jid}/hands?start_ms=1000&end_ms=2000').json()),1)
        self.assertEqual(len(self.client.get(f'/api/clips/{jid}/actions?start_ms=1000').json()['actions']),1)
        srch=self.client.post('/api/search',json={'query':'screwdriver','hand':'right'}).json()
        self.assertEqual(len(srch['results']),1)
        self.assertEqual(srch['results'][0]['clip_id'],jid)
        self.assertEqual(self.client.delete(f'/api/jobs/{jid}').status_code,204)


if __name__=='__main__': unittest.main()
