from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

spec=importlib.util.spec_from_file_location('release_gate',Path(__file__).parents[1]/'scripts/check_release.py')
assert spec is not None and spec.loader is not None
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def accepted(root: Path) -> None:
    (root/'src/video_privacy/assets').mkdir(parents=True)
    (root/'src/video_privacy/__init__.py').write_text('VERSION="fictional"\n')
    (root/'pyproject.toml').write_text('[project]\nname="fictional"\n')
    evidence={'split':'holdout','frames_evaluated':12,'metrics':{
        'holdout_sensitive_core_coverage':1.0,'holdout_sensitive_missed_frames':0,
        'public_background_false_mask_frames':0,'input_method_candidate_false_mask_frames':0,
        'password_transition_failures':0,'notification_transition_failures':0}}
    path=root/'evidence.json';path.write_text(json.dumps(evidence))
    status={'status':'passed','known_failures':[],
            'validated_source_sha256':module.source_digest(root),
            'evidence':[{'path':'evidence.json','sha256':hashlib.sha256(path.read_bytes()).hexdigest()}]}
    (root/'acceptance-status.json').write_text(json.dumps(status))


def test_changed_source_invalidates_previous_acceptance(tmp_path: Path) -> None:
    accepted(tmp_path);assert module.check(tmp_path)==[]
    (tmp_path/'src/video_privacy/__init__.py').write_text('VERSION="changed"\n')
    assert module.check(tmp_path)


def test_failing_evidence_cannot_pass_even_with_matching_hash(tmp_path: Path) -> None:
    accepted(tmp_path)
    path=tmp_path/'evidence.json';data=json.loads(path.read_text())
    data['metrics']['holdout_sensitive_missed_frames']=1
    path.write_text(json.dumps(data))
    status=json.loads((tmp_path/'acceptance-status.json').read_text())
    status['evidence'][0]['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path/'acceptance-status.json').write_text(json.dumps(status))
    assert module.check(tmp_path)


def test_known_failure_blocks_otherwise_valid_acceptance(tmp_path: Path) -> None:
    accepted(tmp_path)
    path=tmp_path/'acceptance-status.json'
    status=json.loads(path.read_text());status['known_failures']=['fictional missed frame']
    path.write_text(json.dumps(status))
    assert module.check(tmp_path)
