#!/bin/sh
set -eu
cd -- "$(dirname -- "$0")"
vpf_python=${VPF_PYTHON:-python3}
"$vpf_python" -c 'import sys; assert (3,11) <= sys.version_info < (3,13), "需要 Python 3.11 或 3.12"'
command -v git >/dev/null 2>&1 || { echo '需要先安裝 Git。' >&2; exit 1; }
"$vpf_python" -m venv .venv
.venv/bin/python scripts/fetch_assets.py
.venv/bin/python -m pip install -c constraints-tested.txt .
printf '\n安裝完成。執行 sh start-web.sh 開啟影片處理頁。\n'
