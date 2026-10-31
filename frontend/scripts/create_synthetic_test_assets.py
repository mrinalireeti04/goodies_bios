"""Create completely synthetic input video and PS results to test the frontend import flow.

No real or private footage is used. Files are written to ignored outputs/synthetic_test/.
Run from repository root: python -m scripts.create_synthetic_test_assets
"""
from __future__ import annotations

import json
from pathlib import Path
from modules.ps4 import run_ps4


def main():
    import cv2
    import numpy as np
    output=Path('outputs/synthetic_test')
    output.mkdir(parents=True,exist_ok=True)
    fps=10
    video=cv2.VideoWriter(str(output/'factory_sample.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(640,360))
    if not video.isOpened():raise RuntimeError('Could not create synthetic video')
    hand_points=[(0,0),(-.32,.14),(-.44,.29),(-.59,.41),(-.66,.5),(-.21,.36),(-.2,.59),(-.2,.77),(-.21,.95),(.04,.39),(.06,.64),(.08,.86),(.1,1.02),(.25,.35),(.3,.59),(.34,.78),(.38,.89),(.43,.21),(.52,.39),(.61,.53),(.68,.64)]
    frames=[];obj_frames=[]
    for i in range(40):
        t=i*100;cx=int(355+20*np.sin(i/7));cy=260
        img=np.full((360,640,3),(41,55,69),dtype=np.uint8)
        cv2.rectangle(img,(90,275),(560,325),(120,134,151),-1)
        cv2.rectangle(img,(275,203),(382,274),(80,101,111),-1)
        cv2.putText(img,'SYNTHETIC TEST FOOTAGE - NOT REAL FACTORY DATA',(35,35),cv2.FONT_HERSHEY_SIMPLEX,.57,(218,228,237),2)
        cv2.putText(img,f't = {t/1000:.1f}s',(35,68),cv2.FONT_HERSHEY_SIMPLEX,.64,(181,191,206),1)
        coords=[[cx+x*47,cy-y*76] for x,y in hand_points]
        for x,y in coords:cv2.circle(img,(int(x),int(y)),3,(67,219,167),-1)
        cv2.rectangle(img,(311,169),(390,229),(72,140,224),2)
        video.write(img)
        kp3=[[x*.05,y*.06,.004*np.sin(j)] for j,(x,y) in enumerate(hand_points)]
        xs=[p[0] for p in coords];ys=[p[1] for p in coords]
        frames.append({'schema_version':'1.0','clip_id':'factory_sample','frame_idx':i,'t_ms':t,'frame_width':640,'frame_height':360,'hands':[{'hand_id':'H0','side':'right','handedness_score':.9,'bbox_px':[min(xs)-5,min(ys)-5,max(xs)+5,max(ys)+5],'kp2d_px':coords,'kp3d_hand_m':kp3,'source':'detected'}]})
        obj_frames.append({'frame_idx':i,'t_ms':t,'objects':[{'object_id':'O0','label':'synthetic tool','type':'portable','bbox_px':[311,169,390,229],'confidence':.84}],'hand_states':[{'hand_id':'H0','state':'portable','object_id':'O0','confidence':.8}]})
    video.release()
    hands_path=output/'hands.jsonl'
    hands_path.write_text('\n'.join(json.dumps(f) for f in frames),encoding='utf-8')
    (output/'ps2.json').write_text(json.dumps({'schema_version':'1.0','clip_id':'factory_sample','frames':obj_frames,'contacts':[{'event_id':'C0','start_ms':500,'end_ms':3500,'hand_id':'H0','object_id':'O0','object_label':'synthetic tool','state':'portable','interaction':'hold','confidence':.8}]},indent=2),encoding='utf-8')
    (output/'actions.json').write_text(json.dumps({'schema_version':'1.0','clip_id':'factory_sample','actions':[{'action_id':'A1','start_ms':0,'end_ms':700,'verb':'pick','object_label':'synthetic tool','object_id':'O0','hand':'right','hand_ids':['H0'],'confidence':.87,'source':'manual','evidence_event_ids':['C0']},{'action_id':'A2','start_ms':900,'end_ms':3300,'verb':'hold','object_label':'synthetic tool','object_id':'O0','hand':'right','hand_ids':['H0'],'confidence':.83,'source':'manual','evidence_event_ids':['C0']},{'action_id':'A3','start_ms':3400,'end_ms':3900,'verb':'place','object_label':'synthetic tool','object_id':'O0','hand':'right','hand_ids':['H0'],'confidence':.8,'source':'manual','evidence_event_ids':['C0']}]},indent=2),encoding='utf-8')
    run_ps4(hands_path,output)
    print(f'Created synthetic video and PS1–PS4 result files in {output.resolve()}')


if __name__=='__main__':main()
