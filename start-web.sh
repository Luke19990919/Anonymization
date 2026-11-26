#!/bin/sh
set -eu
cd -- "$(dirname -- "$0")"
# Optional machine-local runtime pointer; never included in shared archives.
vpf_runtime=.venv/bin/python
if [ -f .local-runtime ]; then IFS= read -r vpf_runtime < .local-runtime; fi
if [ ! -x "$vpf_runtime" ]; then
    sh install.sh
    vpf_runtime=.venv/bin/python
fi
if ! "$vpf_runtime" -c 'import video_privacy; assert video_privacy.__version__ == "0.2.7rc3"' 2>/dev/null; then
    "$vpf_runtime" -m pip install -c constraints-tested.txt .
fi
exec "$vpf_runtime" launch-homepage.py
