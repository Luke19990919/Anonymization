"""Fetch pinned public build assets; never read or transmit user media."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]


def ensure_asset(root: Path,entry: dict[str,str]) -> None:
    destination=(root/entry['path']).resolve()
    if root.resolve() not in destination.parents:raise ValueError('Invalid asset destination')
    if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest()==entry['sha256']:
        return
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary: Path | None=None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent,delete=False) as target:
            temporary=Path(target.name)
            digest=hashlib.sha256()
            with urlopen(entry['url'],timeout=120) as response:
                while chunk:=response.read(1024*1024):
                    digest.update(chunk);target.write(chunk)
        if digest.hexdigest()!=entry['sha256']:raise ValueError('Public asset hash mismatch; file not installed')
        temporary.replace(destination)
        temporary=None
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)


if __name__=='__main__':
    lock=json.loads((ROOT/'assets-lock.json').read_text())
    for entry in lock['assets']:ensure_asset(ROOT,entry)
    print('Pinned public assets verified.')
