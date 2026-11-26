from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

MODEL_REVISION = '7ffa9a043d54d1be65afb281eddf0ffbe629385b'
MODEL_SHA256 = '9c262cbe68a0c8a50590a648ef8341a2b7d3be1fa11dfb79893fe0b03ce57b5c'


def checkpoint_path(models: Path, override: Path | None) -> Path:
    return override if override is not None else models/'opf'/'original'


def check_models(models: Path, checkpoint: Path) -> None:
    for file in ['config.json', 'model.safetensors', 'dtypes.json', 'viterbi_calibration.json']:
        if not (checkpoint/file).is_file():
            raise ValueError('缺少 Privacy Filter 權重；請先執行 vpf prepare-models。')
    if not any((models/'tokenizer').glob('*')):
        raise ValueError('缺少分詞器快取；請先執行 vpf prepare-models。')
    for file in ['craft_mlt_25k.pth', 'chinese.pth']:
        if not (models/'ocr'/file).is_file():
            raise ValueError('缺少繁中 OCR 權重；請先執行 vpf prepare-models。')


def prepare(models: Path, override: Path | None = None) -> dict[str, object]:
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['TIKTOKEN_CACHE_DIR'] = str((models/'tokenizer').resolve())
    models.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_path(models, override)
    if override is None:
        from huggingface_hub import snapshot_download
        snapshot_download('openai/privacy-filter', revision=MODEL_REVISION,
                          allow_patterns=['original/*'], local_dir=models/'opf', token=False)
    with (checkpoint/'model.safetensors').open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    if digest != MODEL_SHA256:
        raise ValueError('Privacy Filter 權重雜湊不符，停止安裝。')
    import tiktoken
    tiktoken.get_encoding('o200k_base')
    import easyocr
    easyocr.Reader(['ch_tra', 'en'], gpu=False, model_storage_directory=str(models/'ocr'),
                   user_network_directory=str(models/'ocr-user'), verbose=False)
    payload = {'model': 'openai/privacy-filter', 'revision': MODEL_REVISION,
               'sha256': digest, 'ocr_languages': ['ch_tra', 'en']}
    (models/'model-receipt.json').write_text(json.dumps(payload, indent=2)+'\n', encoding='utf-8')
    return payload
