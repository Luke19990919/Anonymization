from __future__ import annotations

import math
import os
import gc
from time import perf_counter
from multiprocessing import get_context
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .detection import Box
from .ocr_cache import ExactCropCache


def choose_device(requested: str,mps_available: bool) -> str:
    if requested not in {'auto','cpu','mps'}:
        raise ValueError('VPF_OCR_DEVICE 必須為 auto、cpu 或 mps。')
    return 'mps' if requested!='cpu' and mps_available else 'cpu'


def _worker(pipe: Connection, models: str, languages: tuple[str,...]) -> None:
    # Keep OpenCV's native libraries out of the PyAV process, especially on macOS.
    try:
        import easyocr
        import easyocr.easyocr as engine
        import torch
        root = Path(models)
        device=choose_device(os.environ.get('VPF_OCR_DEVICE','auto'),torch.backends.mps.is_available())
        fallbacks=0
        def create_reader(selected: str) -> Any:
            return easyocr.Reader(list(languages),gpu='mps' if selected=='mps' else False,
                                  model_storage_directory=str(root/'ocr'),
                                  user_network_directory=str(root/'ocr-user'),
                                  download_enabled=False,verbose=False)
        try:
            reader=create_reader(device)
        except Exception:
            if device!='mps':raise
            device='cpu';fallbacks+=1
            reader=create_reader(device)
        stage_seconds: dict[str,float] = {'ocr_detection_seconds':0.0,'ocr_recognition_seconds':0.0}
        def timed_method(method: Any,key: str) -> Any:
            def measured(*args: Any,**kwargs: Any) -> Any:
                started=perf_counter()
                try:
                    return method(*args,**kwargs)
                finally:
                    stage_seconds[key]+=perf_counter()-started
            return measured
        def instrument(selected_reader: Any) -> None:
            selected_reader.detect=timed_method(selected_reader.detect,'ocr_detection_seconds')
            selected_reader.recognize=timed_method(selected_reader.recognize,'ocr_recognition_seconds')
        instrument(reader)
        crop_cache=ExactCropCache(engine.get_text)
        engine.get_text=crop_cache
        pipe.send(('ready',None))
        while True:
            rgb = pipe.recv()
            if rgb is None:
                break
            height, width = rgb.shape[:2]
            try:
                raw=reader.readtext(rgb,detail=1,paragraph=False,workers=0,
                                    text_threshold=.5,low_text=.3,min_size=0)
            except Exception:
                if device!='mps':raise
                # Retry this same frame on CPU; never skip a failed GPU frame.
                reader=None
                gc.collect()
                torch.mps.empty_cache()
                crop_cache.cache.clear()
                device='cpu';fallbacks+=1
                reader=create_reader(device)
                instrument(reader)
                raw=reader.readtext(rgb,detail=1,paragraph=False,workers=0,
                                    text_threshold=.5,low_text=.3,min_size=0)
            boxes: list[Box] = []
            for points,text,confidence in raw:
                x = [float(p[0]) for p in points]; y = [float(p[1]) for p in points]
                rect = (max(0,math.floor(min(x))), max(0,math.floor(min(y))),
                        min(width,math.ceil(max(x))), min(height,math.ceil(max(y))))
                if rect[2]>rect[0] and rect[3]>rect[1]:
                    boxes.append(Box(rect,str(text),float(confidence)))
            pipe.send(('result',(boxes,{'ocr_device':device,'ocr_crop_cache_hits':crop_cache.hits,
                                        'ocr_crop_cache_misses':crop_cache.misses,
                                        'ocr_cpu_fallbacks':fallbacks,**stage_seconds})))
    except (EOFError,BrokenPipeError):
        pass
    except Exception:
        # No OCR text or library exception payload crosses into logs.
        try:
            pipe.send(('error',None))
        except (EOFError,BrokenPipeError,OSError):
            pass
    finally:
        pipe.close()


class EasyOCRReader:
    def __init__(self, models: Path, config: Config) -> None:
        context = get_context('spawn')
        self.statistics: dict[str,object]={}
        self.pipe, child = context.Pipe()
        self.process = context.Process(target=_worker,args=(child,str(models),config.languages),daemon=True)
        self.process.start()
        child.close()
        try:
            state,_ = self._receive()
            if state != 'ready':
                raise RuntimeError('OCR 模型載入失敗，請重新準備模型。')
        except BaseException:
            self.close()
            raise

    def _receive(self) -> tuple[str,Any]:
        if not self.pipe.poll(600):
            raise RuntimeError('OCR 處理超時；已停止輸出，未略過畫面。')
        try:
            return self.pipe.recv()
        except (EOFError,OSError) as exc:
            raise RuntimeError('OCR 子程序中斷；已停止輸出。') from exc

    def read(self, rgb: np.ndarray) -> list[Box]:
        self.pipe.send(rgb)
        state,boxes = self._receive()
        if state != 'result':
            raise RuntimeError('OCR 偵測失敗；已停止輸出，未略過畫面。')
        found,statistics=boxes
        self.statistics=statistics
        return found

    def summary(self) -> dict[str,object]:
        return dict(self.statistics)

    def close(self) -> None:
        try:
            if self.process.is_alive():
                self.pipe.send(None)
        except (EOFError,BrokenPipeError,OSError):
            pass
        self.process.join(timeout=3)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=3)
        self.pipe.close()

    def __enter__(self) -> EasyOCRReader:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
