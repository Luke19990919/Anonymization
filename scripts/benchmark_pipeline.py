"""Local paired diagnostic. Input and outputs stay in the supplied private directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
from time import perf_counter

from video_privacy.detection import Mask
from video_privacy.processing import run_automatic


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('input',type=Path)
    parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False,mode=0o700)
    masks: list[list[tuple[int,list[Mask]]]]=[]
    runs=[]
    for enabled in (False,True):
        observed: list[tuple[int,list[Mask]]]=[]
        started=perf_counter()
        report=run_automatic(args.input,args.output_dir/f'prefetch-{enabled}.mp4',args.models,
                             observer=lambda i,regions:observed.append((i,list(regions))),
                             prefetch_ocr=enabled)
        masks.append(observed)
        runs.append({'prefetch':enabled,'elapsed_including_models_seconds':perf_counter()-started,
                     'frames':report['frames_processed'],'performance':report['performance'],
                     **{k:v for k,v in report.items() if k.startswith(('nlp_','ocr_'))}})
    result={'machine':platform.machine(),'system':platform.system(),'runs':runs,
            'all_frame_masks_equal':masks[0]==masks[1],
            'scope':'paired local diagnostic; same input; no gold; not privacy accuracy acceptance',
            'raw_text_saved':False}
    (args.output_dir/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
