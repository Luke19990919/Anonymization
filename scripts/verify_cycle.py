"""Repeatable local validation cycle; repair execution is intentionally a host responsibility."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from check_release import check,source_digest

ROOT=Path(__file__).resolve().parents[1]


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument('--models',type=Path,required=True)
    args=parser.parse_args()
    folder=ROOT/'.runtime'/'verify-cycles'/uuid.uuid4().hex
    folder.mkdir(parents=True,mode=0o700)
    environment=dict(os.environ)
    environment['PYTHONPATH']=str(ROOT/'src')
    commands=[('python_regression',[sys.executable,'-m','pytest','-q']),
              ('eta_regression',['node','tests/test_eta.cjs']),
              ('generate_fictional_fixture',[sys.executable,'examples/regression-stacked/generate.py',str(folder/'fixture')]),
              ('actual_ocr_nlp_regression',[sys.executable,'scripts/validate_stacked.py',str(folder/'fixture'),
                                            '--models',str(args.models.resolve())]),
              ('generate_notification_fixture',[sys.executable,'examples/regression-notification/generate.py',str(folder/'notifications')]),
              ('actual_notification_regression',[sys.executable,'scripts/validate_stacked.py',str(folder/'notifications'),
                                                  '--models',str(args.models.resolve()),'--scope','fictional branded notification fixture'])]
    stages=[]
    for name,command in commands:
        with (folder/f'{name}.log').open('wb') as log:
            result=subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,check=False)
        stages.append({'name':name,'passed':result.returncode==0,'exit_code':result.returncode})
        print(json.dumps(stages[-1]),flush=True)
        if result.returncode!=0:break
    release_errors=check(ROOT)
    report={'schema_version':1,'source_sha256':source_digest(ROOT),'stages':stages,
            'regression_passed':len(stages)==len(commands) and all(stage['passed'] for stage in stages),
            'release_allowed':not release_errors,'release_errors':release_errors,
            'repair_executor_connected':False,'private_media_uploaded':False}
    (folder/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return int(not report['regression_passed'] or not report['release_allowed'])


if __name__=='__main__':
    raise SystemExit(main())
