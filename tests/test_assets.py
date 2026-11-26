from __future__ import annotations

import hashlib
import importlib.util
import io
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('fetch_assets',Path(__file__).parents[1]/'scripts/fetch_assets.py')
assert spec and spec.loader
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_wrong_asset_hash_never_installs_partial_data(tmp_path: Path,monkeypatch) -> None:
    monkeypatch.setattr(module,'urlopen',lambda *a,**k:io.BytesIO(b'fictional corrupted asset'))
    entry={'path':'assets/font.ttf','url':'https://example.invalid/font','sha256':'0'*64}
    with pytest.raises(ValueError,match='hash mismatch'):module.ensure_asset(tmp_path,entry)
    assert list((tmp_path/'assets').iterdir())==[]


def test_verified_asset_does_not_need_network(tmp_path: Path,monkeypatch) -> None:
    path=tmp_path/'font.ttf';path.write_bytes(b'fictional verified asset')
    def forbidden(*args,**kwargs):raise AssertionError('network should not run')
    monkeypatch.setattr(module,'urlopen',forbidden)
    module.ensure_asset(tmp_path,{'path':'font.ttf','url':'https://example.invalid/font',
                                  'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
