import unittest
from unittest.mock import patch
from fastapi import HTTPException
from backend import app


class ManualMaskTests(unittest.TestCase):
    def test_endpoint_returns_frame_and_mask(self):
        job={'job_id':'fixture','media_type':'video','path':'example.mp4','status':'completed','module_status':{}}
        request=app.ManualMaskRequest(frame_idx=10,bbox_px=[1,2,30,40])
        polygon=[[1,2],[30,2],[30,40]]
        with patch.dict(app.JOBS,{'fixture':job},clear=True), patch('backend.inference.manual_mask.segment_frame',return_value=polygon):
            result=app.manual_mask('fixture',request)
        self.assertEqual(result,{'frame_idx':10,'mask_polygon_px':polygon})

    def test_active_processing_blocks_mask_generation(self):
        job={'job_id':'fixture','media_type':'video','path':'example.mp4','status':'processing','module_status':{}}
        with patch.dict(app.JOBS,{'fixture':job},clear=True):
            with self.assertRaises(HTTPException) as error:
                app.manual_mask('fixture',app.ManualMaskRequest(frame_idx=10,bbox_px=[1,2,30,40]))
        self.assertEqual(error.exception.status_code,409)
