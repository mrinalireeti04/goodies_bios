import json
import tempfile
from types import SimpleNamespace
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from backend import app


class JobAuditTests(unittest.TestCase):
    def test_module_timing_records_duration_and_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            job={'directory':Path(directory),'job_id':'test','module_status':{},'completed_modules':[],'issues':[]}
            with patch.dict(app.JOBS,{'test':job}):
                app.update_module('test','ps2','processing')
                self.assertIn('started_at',job['module_timing']['ps2'])
                time.sleep(.01)
                app.update_module('test','ps2','completed',completed=True)
                self.assertGreaterEqual(job['module_timing']['ps2']['duration_s'],0)
                self.assertIn('ps2',job['completed_modules'])

    def test_job_persists_and_exposes_total_and_stage_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            job={'directory':root,'path':root/'input.mp4','job_id':'audit-test','media_type':'video',
                 'module_status':{},'completed_modules':[],'issues':[], 'enable_ps2':False,'enable_ps3':False}
            def run_ps1(video, out):
                (out/'ps4.json').write_text('{}')
                hands=out/'hands.jsonl'
                hands.write_text('')
                return hands
            with patch.dict(app.JOBS,{'audit-test':job}), patch.object(app.importlib,'import_module',return_value=SimpleNamespace(run_ps1=run_ps1)):
                app.process_job('audit-test')
                public=app.get_status('audit-test')
                saved=json.loads((root/'job.json').read_text())
            for data in (public,saved):
                self.assertEqual(data['status'],'completed')
                self.assertGreaterEqual(data['processing_duration_s'],0)
                self.assertGreaterEqual(data['module_timing']['ps1']['duration_s'],0)
                self.assertEqual(data['module_status']['ps2'],'skipped')


if __name__=='__main__':
    unittest.main()
