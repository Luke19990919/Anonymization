"""Experimental macOS Vision OCR; RGB and recognized text travel through local pipes only."""
from __future__ import annotations

import json
from pathlib import Path
import select
import struct
import subprocess
from time import perf_counter

import numpy as np

from .detection import Box


class VisionOCRReader:
    def __init__(self,executable: Path) -> None:
        self.process=subprocess.Popen([str(executable.resolve())],stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
        self.frames=0
        self.seconds=0.0

    def read(self,rgb: np.ndarray) -> list[Box]:
        if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:
            raise ValueError('OCR requires RGB uint8 pixels')
        started=perf_counter()
        height,width=rgb.shape[:2]
        assert self.process.stdin is not None and self.process.stdout is not None
        try:
            self.process.stdin.write(struct.pack('<II',width,height))
            self.process.stdin.write(np.ascontiguousarray(rgb).tobytes())
            self.process.stdin.flush()
            if not select.select([self.process.stdout],[],[],600)[0]:
                raise RuntimeError('OCR 處理超時；停止輸出，未略過畫面。')
            raw=json.loads(self.process.stdout.readline())
            boxes=[]
            for item in raw:
                rect=tuple(int(value) for value in item['rect'])
                if len(rect)!=4 or not (0<=rect[0]<rect[2]<=width and 0<=rect[1]<rect[3]<=height):
                    raise ValueError('Invalid OCR geometry')
                boxes.append(Box(rect,str(item['text']),float(item['confidence'])))
            self.frames+=1
            return boxes
        except (OSError,ValueError,KeyError,TypeError) as exc:
            raise RuntimeError('Vision OCR 失敗；停止輸出，未略過畫面。') from exc
        finally:
            self.seconds+=perf_counter()-started

    def summary(self) -> dict[str,object]:
        return {'ocr_backend':'apple_vision_experimental','ocr_frames':self.frames,
                'ocr_total_seconds':self.seconds,'ocr_device':'system_managed'}

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=3)
        if self.process.stdout is not None:
            self.process.stdout.close()

    def __enter__(self) -> VisionOCRReader:
        return self

    def __exit__(self,*args: object) -> None:
        self.close()
