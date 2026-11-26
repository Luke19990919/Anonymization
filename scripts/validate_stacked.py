"""Actual-model fixture evaluation. Gold only enters the observer, never detection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import av

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
    misses=[];target_instances=0;empty_mask_frames=[];unexpected_masks={}
    def observe(index: int,masks: list[Mask]) -> None:
        nonlocal target_instances
        targets=truth['frames'][index]['targets']
        if not targets and masks:
            empty_mask_frames.append(index);unexpected_masks[index]=list(masks)
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
    empty_diagnostics=[]
    if unexpected_masks:
        with av.open(str(args.folder/'input.mp4')) as original:
            for index,frame in enumerate(original.decode(video=0)):
                if index not in unexpected_masks:continue
                rgb=frame.to_ndarray(format='rgb24')
                for mask in unexpected_masks[index]:
                    x1,y1,x2,y2=mask.rect;patch=rgb[y1:y2,x1:x2]
                    if not patch.size:continue
                    colors,counts=np.unique(patch.reshape(-1,3),axis=0,return_counts=True)
                    empty_diagnostics.append({'frame':index,'reason':mask.reason,'rect':list(mask.rect),
                        'min_rgb':colors.min(axis=0).tolist(),'max_rgb':colors.max(axis=0).tolist(),
                        'unique_colors':len(colors),'nonmodal_pixels':int(counts.sum()-counts.max())})
    result={'program_version' :report['program_version'],'frames':report['frames_processed'],
            'target_instances':target_instances,'misses':misses,'empty_mask_frames':empty_mask_frames,
            'gold_passed_to_detector':False,'actual_model_inference':True,
            'scope':args.scope,
            'performance':report['performance'],'empty_mask_diagnostics':empty_diagnostics,
            'ocr_statistics':{key:report[key] for key in ('ocr_device','ocr_cpu_fallbacks','ocr_crop_cache_hits','ocr_crop_cache_misses') if key in report},
            'mask_applications_by_reason':report['mask_applications_by_reason']}
    (args.folder/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    raise SystemExit(bool(misses or empty_mask_frames))


if __name__=='__main__':
    main()
