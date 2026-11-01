import unittest
from backend.inference.ps3.action_recognizer import build_coarse_prompt, recognize_actions_full_video
from backend.inference.ps3.schemas import VideoSegment


class ObjectContextTests(unittest.TestCase):
    def test_context_is_bounded_to_segment_and_untrusted(self):
        segment=VideoSegment(index=1,start_s=100,end_s=150,duration_s=50)
        context=[{'object':'aluminum profile','start_s':105,'end_s':110,'confidence':.8},
                 {'object':'outside','start_s':0,'end_s':20,'confidence':.9},
                 {'object':'weak','start_s':105,'end_s':110,'confidence':.2}]
        prompt=build_coarse_prompt(segment,context)
        self.assertIn('aluminum profile',prompt)
        self.assertIn('"start_s": 5.0',prompt)
        self.assertNotIn('"object": "outside"',prompt)
        self.assertNotIn('"object": "weak"',prompt)
        self.assertIn('Unverified detector suggestions',prompt)

    def test_context_reaches_segment_recognition(self):
        class FakeClient:
            def __init__(self): self.prompts=[]
            def analyze_video(self,video,prompt,json_mode=False): self.prompts.append(prompt); return '[]'
            def parse_json_response(self,text): return []
        client=FakeClient()
        segment=VideoSegment(index=0,start_s=0,end_s=20,duration_s=20)
        recognize_actions_full_video(client,{0:'video'},[segment],object_context=[{'object':'wrench','start_s':2,'end_s':5,'confidence':.9}])
        self.assertEqual(len(client.prompts),1)
        self.assertIn('wrench',client.prompts[0])


if __name__=='__main__':
    unittest.main()
