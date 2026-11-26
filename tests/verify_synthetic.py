"""Pixel and report checks for the bundled, explicitly fictional example."""
from __future__ import annotations

import argparse
from itertools import islice
import json
from pathlib import Path

import av
import numpy as np


def frames(path: Path) -> list[np.ndarray]:
    with av.open(str(path)) as video:
        return [frame.to_ndarray(format='rgb24') for frame in islice(video.decode(video=0),13)]


def verify(folder: Path) -> dict[str,object]:
    truth=json.loads((folder/'expected.synthetic.json').read_text())
    assert truth['synthetic'] is True and truth['frames']==12
    before=frames(folder/'synthetic-input.mp4')
    after=frames(folder/'redacted-output.mp4')
    assert len(before)==len(after)==12
    regions={'notification':(620,40,920,150), 'password':(40,435,900,480),
             'email':(45,230,410,265), 'id':(45,300,200,330),
             'address':(45,370,370,400), 'model_only_english_person':(48,525,260,545)}
    blur_checks: dict[str,dict[str,float]]={}
    for label,(x1,y1,x2,y2) in regions.items():
        original=[frame[y1:y2,x1:x2].astype(float) for frame in before]
        patches=[frame[y1:y2,x1:x2].astype(float) for frame in after]
        def edges(patch: np.ndarray) -> float:
            return float(np.abs(np.diff(patch,axis=0)).mean()+np.abs(np.diff(patch,axis=1)).mean())
        pairs=[(edges(a),edges(b)) for a,b in zip(original,patches) if edges(a)>.5]
        assert pairs, f'{label} source fixture has no visible detail'
        ratio=max(b/a for a,b in pairs)
        assert ratio<.4, f'{label} blur detail suppression failed'
        blur_checks[label]={'max_edge_strength_ratio':ratio}
    footer_errors=[float(np.abs(a[575:615,35:800].astype(float)-b[575:615,35:800]).mean())
                   for a,b in zip(before,after)]
    assert max(footer_errors)<4
    header_errors=[float(np.abs(a[30:80,35:380].astype(float)-b[30:80,35:380]).mean())
                   for a,b in zip(before,after)]
    assert max(header_errors)<4
    aliases=[frame[128:192,28:240] for frame in after]
    same_person_error=float(np.abs(aliases[0].astype(float)-aliases[8]).mean())
    different_person_error=float(np.abs(aliases[0].astype(float)-aliases[4]).mean())
    assert same_person_error<3 and different_person_error>.1
    report_text=(folder/'redacted-output.report.json').read_text()
    for secret in ['study.user@example.invalid','A123456789','臺北市','Alice Smith','王小明','陳小華','DemoOnly']:
        assert secret not in report_text
    report=json.loads(report_text)
    assert report['mask_applications_by_reason'].get('model',0)>0
    assert report['mask_applications_by_reason']['participant_region']==12
    assert report['mask_style']=='gaussian_blur'
    assert report['blur_uses_original_pixels'] is True
    assert report['irreversible_redaction_guaranteed'] is False
    result={'scope':'bundled synthetic clip only; not a detection accuracy benchmark',
            'frames':12, 'notification_single_frame_covered':True, 'password_single_frame_covered':True,
            'gaussian_blur_regions':blur_checks, 'retained_footer_max_mean_error':max(footer_errors),
            'retained_header_max_mean_error':max(header_errors),
            'same_participant_return_error':same_person_error,
            'different_participant_pixel_difference':different_person_error,
            'model_only_person_mask_applied':True, 'raw_sensitive_text_absent_from_report':True}
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('folder',type=Path)
    args=parser.parse_args()
    print(json.dumps(verify(args.folder),ensure_ascii=False,indent=2))
