"""Allowlisted bundles; known privacy failures permit candidate archives only."""
from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

from check_release import check
from video_privacy import __version__

ROOT=Path(__file__).resolve().parents[1]
FILES=['assets-lock.json','PROJECT_STATUS.md','README.md','ADVANCED.md','THIRD_PARTY_NOTICES.md','VALIDATION.md','HANDOFF.md',
       'CICD.md','MAINTENANCE.md','OPEN_SOURCE_REVIEW.md','MANIFEST.in','pyproject.toml','constraints-tested.txt',
       'release-manifest.json','acceptance-status.json','install.sh','install.ps1',
       'start-web.sh','launch-homepage.py','啟動影片處理.command','啟動影片處理.bat']
TREES=['src/video_privacy','tests','scripts','.github/workflows','examples','templates','schemas']


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',action='store_true')
    args=parser.parse_args()
    if not args.candidate:
        failures=check(ROOT)
        if failures:raise SystemExit('\n'.join(failures))
    label=f'影片隱私處理-{__version__}'+('-待驗收' if args.candidate else '-LINE分享包')
    archive=ROOT/f'{label}.zip'
    paths={ROOT/name for name in FILES}
    for folder in TREES:
        paths.update(path for path in (ROOT/folder).rglob('*') if path.is_file() and
                     '__pycache__' not in path.parts and path.suffix not in {'.pyc','.pyo'})
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(paths):
            bundle.write(path,f'{label}/{path.relative_to(ROOT).as_posix()}')
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        assert not any(any(part in name.split('/') for part in
                       ('private-results','models','.venv','.runtime','.local-runtime','.git'))
                       for name in bundle.namelist())
    print(archive)
