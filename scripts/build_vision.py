"""Build the optional macOS OCR helper into the excluded local runtime directory."""
from __future__ import annotations

from pathlib import Path
import platform
import subprocess

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    if platform.system()!='Darwin':raise SystemExit('Apple Vision 測試後端僅適用 macOS。')
    runtime=ROOT/'.runtime'
    runtime.mkdir(exist_ok=True,mode=0o700)
    subprocess.run(['xcrun','swiftc','-O','-module-cache-path',str(runtime/'swift-cache'),
                    str(ROOT/'src/video_privacy/assets/vision-ocr.swift'),
                    '-o',str(runtime/'vision-ocr')],check=True)
    print(runtime/'vision-ocr')
