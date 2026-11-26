"""Fail closed until independent privacy acceptance exists for the exact source."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def source_digest(root: Path) -> str:
    digest=hashlib.sha256()
    paths=[*root.joinpath('src/video_privacy').rglob('*.py'),
           *root.joinpath('src/video_privacy/assets').glob('*'),
           root/'pyproject.toml']
    for path in sorted(set(paths)):
        if not path.is_file():continue
        digest.update(path.relative_to(root).as_posix().encode()+b'\0')
        digest.update(path.read_bytes())
    return digest.hexdigest()


def check(root: Path) -> list[str]:
    status=json.loads((root/'acceptance-status.json').read_text())
    errors=[]
    if status.get('status')!='passed':errors.append('隱私驗收尚未通過。')
    if status.get('known_failures'):errors.append('仍有已知漏遮或誤遮。')
    if status.get('validated_source_sha256')!=source_digest(root):errors.append('驗收證據未對應目前程式。')
    evidence=status.get('evidence',[])
    if not evidence:errors.append('缺少獨立保留集驗收證據。')
    for entry in evidence:
        path=root/entry['path']
        if root.resolve() not in path.resolve().parents or not path.is_file():
            errors.append('驗收證據檔案無效。');continue
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry.get('sha256'):
            errors.append('驗收證據雜湊不符。');continue
        measurement=json.loads(path.read_text())
        if measurement.get('split')!='holdout' or measurement.get('frames_evaluated',0)<1:
            errors.append('證據不是有效的獨立保留集。')
        expected={'holdout_sensitive_core_coverage':1.0,'holdout_sensitive_missed_frames':0,
                  'public_background_false_mask_frames':0,'input_method_candidate_false_mask_frames':0,
                  'password_transition_failures':0,'notification_transition_failures':0}
        if any(measurement.get('metrics',{}).get(key)!=value for key,value in expected.items()):
            errors.append('漏遮、誤遮或切換邊界驗收未達標。')
    return errors


if __name__=='__main__':
    errors=check(ROOT)
    print('\n'.join(errors) if errors else '發布驗收閘門通過。')
    sys.exit(bool(errors))
