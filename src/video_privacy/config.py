from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any
import unicodedata


def normalize(text: str) -> str:
    return ''.join(unicodedata.normalize('NFKC', text).casefold().split())


def letter_id(index: int) -> str:
    result = ''
    index += 1
    while index:
        index, remainder = divmod(index-1, 26)
        result = chr(65+remainder)+result
    return result


@dataclass(frozen=True)
class Participant:
    aliases: tuple[str, ...]
    replacement: str


@dataclass(frozen=True)
class Region:
    name: str
    kind: str
    rect: tuple[float, float, float, float]
    start: float = 0.0
    end: float | None = None
    replacement: str | None = None

    def active(self, seconds: float) -> bool:
        return seconds >= self.start and (self.end is None or seconds < self.end)

    def pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.rect
        return (math.floor(x1*width), math.floor(y1*height),
                math.ceil(x2*width), math.ceil(y2*height))


@dataclass(frozen=True)
class Config:
    participants: tuple[Participant, ...]
    sensitive_literals: tuple[str, ...]
    regions: tuple[Region, ...]
    padding: int = 8
    low_confidence: float = 0.3
    languages: tuple[str, ...] = ('ch_tra', 'en')


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('座標、時間與門檻必須是有限數值。')
    return float(value)


def _strings(value: object, *, nonempty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(v, str) or not v.strip() for v in value):
        raise ValueError('文字清單必須由非空字串組成。')
    if nonempty and not value:
        raise ValueError('每位受測者至少需要一個姓名或別名。')
    return tuple(value)


def parse_config(data: dict[str, Any]) -> Config:
    allowed = {'schema_version', 'participants', 'sensitive_literals', 'regions',
               'padding', 'low_confidence', 'languages'}
    if set(data)-allowed or data.get('schema_version') != 1:
        raise ValueError('設定版本或欄位不正確；請使用隨附的設定工具。')
    participants: list[Participant] = []
    seen: set[str] = set()
    replacements: set[str] = set()
    entries = data.get('participants', [])
    if not isinstance(entries, list):
        raise ValueError('participants 必須是清單。')
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry)-{'aliases', 'replacement'}:
            raise ValueError('人物欄位不正確。')
        aliases = _strings(entry.get('aliases'), nonempty=True)
        replacement = entry.get('replacement', f'受測者 {letter_id(index)}')
        if not isinstance(replacement, str) or not replacement.strip() or len(replacement) > 40:
            raise ValueError('人物代號必須為 1 至 40 個字元。')
        if replacement in replacements:
            raise ValueError('不同人物不能共用同一個代號。')
        replacements.add(replacement)
        for alias in aliases:
            normalized = normalize(alias)
            if not normalized or normalized in seen:
                raise ValueError('人物姓名或別名重複，無法建立唯一對照。')
            seen.add(normalized)
        participants.append(Participant(aliases, replacement))
    regions: list[Region] = []
    raw_regions = data.get('regions', [])
    if not isinstance(raw_regions, list):
        raise ValueError('regions 必須是清單。')
    for entry in raw_regions:
        if not isinstance(entry, dict) or set(entry)-{'name', 'kind', 'rect', 'start', 'end', 'replacement'}:
            raise ValueError('區域欄位不正確。')
        rect = entry.get('rect')
        if not isinstance(rect, list) or len(rect) != 4:
            raise ValueError('rect 必須為 [左, 上, 右, 下] 四個比例值。')
        coords = tuple(_number(v) for v in rect)
        x1, y1, x2, y2 = coords
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError('區域比例值必須介於 0 與 1，且右下角必須大於左上角。')
        start = _number(entry.get('start', 0))
        end = None if entry.get('end') is None else _number(entry['end'])
        if start < 0 or (end is not None and end <= start):
            raise ValueError('時間必須非負，結束時間必須晚於開始時間。')
        name = entry.get('name', 'mask')
        kind = entry.get('kind', 'manual')
        if not isinstance(name, str) or not name or kind not in {'manual', 'notification', 'participant'}:
            raise ValueError('區域名稱或類型不正確。')
        replacement = entry.get('replacement')
        if kind == 'participant':
            if replacement not in replacements:
                raise ValueError('姓名區域的代號必須與人物清單中的代號完全相同。')
        elif replacement is not None:
            raise ValueError('只有姓名代號區域可以指定 replacement。')
        regions.append(Region(name, kind, coords, start, end, replacement))
    padding = data.get('padding', 8)
    if isinstance(padding, bool) or not isinstance(padding, int) or not 0 <= padding <= 100:
        raise ValueError('padding 必須為 0 至 100 的整數。')
    confidence = _number(data.get('low_confidence', 0.3))
    if not 0 <= confidence <= 1:
        raise ValueError('low_confidence 必須介於 0 與 1。')
    languages = _strings(data.get('languages', ['ch_tra', 'en']), nonempty=True)
    if languages != ('ch_tra', 'en'):
        raise ValueError('此版本固定使用繁體中文與英文 OCR。')
    return Config(tuple(participants), _strings(data.get('sensitive_literals', [])),
                  tuple(regions), padding, confidence, languages)


def load_config(path: Path) -> Config:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('設定必須為 JSON 物件。')
    return parse_config(data)
