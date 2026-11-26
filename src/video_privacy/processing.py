from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from .automatic import AutomaticDetector
from .config import Config
from .detection import Mask, PrivacyClassifier
from .models import check_models, checkpoint_path
from .ocr import EasyOCRReader


def run_automatic(source: Path,destination: Path,models: Path,checkpoint: Path | None=None,
                  progress: Callable[[int],None] | None=None,
                  observer: Callable[[int,list[Mask]],None] | None=None,
                  report_path: Path | None=None, *, prefetch_ocr: bool=True,
                  experimental_search_context: bool=False,
                  vision_executable: Path | None=None) -> dict[str,object]:
    from .video import process_video
    models=models.resolve()
    weights=checkpoint_path(models,checkpoint)
    check_models(models,weights)
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
    os.environ['TIKTOKEN_CACHE_DIR']=str(models/'tokenizer')
    config=Config((),(),())
    detector=AutomaticDetector(PrivacyClassifier(str(weights)),
                               experimental_search_context=experimental_search_context)
    if vision_executable is None:
        reader=EasyOCRReader(models,config)
    else:
        from .vision_ocr import VisionOCRReader
        reader=VisionOCRReader(vision_executable)
    with reader as ocr:
        return process_video(source,destination,report_path or destination.with_suffix('.report.json'),config,
                             detector,ocr,'drop',progress,observer,prefetch_ocr=prefetch_ocr)
