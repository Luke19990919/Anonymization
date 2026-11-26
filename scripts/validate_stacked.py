"""Actual-model fixture evaluation. Gold only enters the observer, never detection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from video_privacy.detection import Mask
from video_privacy.processing import run_automatic


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('folder',type=Path)
    parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--vision-executable',type=Path)
    parser.add_argument('--scope',default='fictional stacked phone/email fixture; not unknown-video acceptance')
    args=parser.parse_args()
    truth=json.loads((args.folder/'gold.json').read_text())
    assert truth['synthetic'] is True
    misses=[];target_instances=0;empty_mask_frames=[]
    def observe(index: int,masks: list[Mask]) -> None:
        nonlocal target_instances
        targets=truth['frames'][index]['targets']
        if not targets and masks:empty_mask_frames.append(index)
        for x1,y1,x2,y2 in targets:
            target_instances+=1
            core=np.zeros((y2-y1,x2-x1),dtype=bool)
            for mask in masks:
                a,b,c,d=mask.rect
                left,top,right,bottom=max(a,x1),max(b,y1),min(c,x2),min(d,y2)
                if left<right and top<bottom:core[top-y1:bottom-y1,left-x1:right-x1]=True
            if not core.all():misses.append({'frame':index,'coverage':float(core.mean())})
    report=run_automatic(args.folder/'input.mp4',args.folder/'output.mp4',args.models,
                         vision_executable=args.vision_executable,observer=observe,
                         progress=lambda count:print(f'frames {count}',flush=True) if count%10==0 else None)
    result={'program_version':report['program_version'],'frames':report['frames_processed'],
            'target_instances':target_instances,'misses':misses,'empty_mask_frames':empty_mask_frames,
            'gold_passed_to_detector':False,'actual_model_inference':True,
            'scope':args.scope,
            'performance':report['performance']}
    (args.folder/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    raise SystemExit(bool(misses or empty_mask_frames))


if __name__=='__main__':
    main()
