from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import re
from time import perf_counter
from typing import Protocol

from .config import Config, normalize


@dataclass(frozen=True)
class Box:
    rect: tuple[int, int, int, int]
    text: str
    confidence: float


@dataclass(frozen=True)
class Mask:
    rect: tuple[int, int, int, int]
    reason: str
    replacement: str | None = None


class TextClassifier(Protocol):
    def spans(self, text: str) -> list[tuple[int, int, str]]: ...


class PrivacyClassifier:
    def __init__(self, checkpoint: str) -> None:
        from opf import OPF
        self.model = OPF(model=checkpoint, device='cpu', context_window_length=1024)
        self.inference_seconds = 0.0
        self.inference_calls = 0
        self.cache: OrderedDict[str, list[tuple[int, int, str]]] = OrderedDict()

    def spans(self, text: str) -> list[tuple[int, int, str]]:
        if not text.strip():
            return []
        if text in self.cache:
            self.cache.move_to_end(text)
            return self.cache[text]
        started = perf_counter()
        try:
            result = self.model.redact(text)
        finally:
            self.inference_seconds += perf_counter()-started
            self.inference_calls += 1
        if result.warning is not None:
            # Mapping cannot be trusted; mask every OCR box touching the full text.
            found = [(0, len(text), 'encoding_mismatch')]
        else:
            found = [(s.start, s.end, s.label) for s in result.detected_spans]
        self.cache[text] = found
        if len(self.cache) > 512:
            self.cache.popitem(last=False)
        return found


EMAIL = re.compile(r'[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}', re.I)
TW_ID = re.compile(r'(?<![A-Z0-9])(?:[A-Z][0-9]{9}|[A-Z]{2}[0-9]{8})(?![A-Z0-9])', re.I)
PHONE = re.compile(r'(?:\+?886[- ]?|0)9[0-9][- ]?[0-9]{3}[- ]?[0-9]{3}|\b\+?[0-9][0-9 ()-]{7,}[0-9]\b')
ADDRESS = re.compile(r'(?:[\u3400-\u9fff]{1,6}[縣市])[^\n]{0,40}(?:路|街|巷|弄)[^\n]{0,20}[0-9]+號')
CUES = re.compile(r'帳號|帳戶|密碼|住址|地址|身分證|身份證|password|passcode|username|user\s*name|account|address', re.I)


class Detector:
    def __init__(self, config: Config, classifier: TextClassifier) -> None:
        self.config = config
        self.classifier = classifier

    def masks(self, boxes: list[Box], width: int, height: int) -> list[Mask]:
        text = '\n'.join(box.text for box in boxes)
        try:
            spans = self.classifier.spans(text)
        except Exception as exc:
            raise RuntimeError('文字模型推論失敗；停止輸出，未略過任何畫面。') from exc
        result: list[Mask] = []
        normalized_parts = [normalize(box.text) for box in boxes]
        normalized_text = ''
        ranges: list[tuple[int,int]] = []
        position = 0
        for index,part in enumerate(normalized_parts):
            if index:
                before,current=boxes[index-1].rect,boxes[index].rect
                overlap=max(0,min(before[3],current[3])-max(before[1],current[1]))
                line_height=min(before[3]-before[1],current[3]-current[1])
                adjacent=(overlap>=line_height*.5 and before[2]-8<=current[0]
                          and current[0]-before[2]<=max(40,line_height*3))
                if not adjacent:
                    normalized_text+='\n';position+=1
            ranges.append((position, position+len(part)))
            normalized_text+=part
            position += len(part)
        sensitive_indexes: set[int] = set()
        name_groups: dict[tuple[int,int,int,int], list[str]] = {}

        def affected(start: int, end: int) -> list[int]:
            return [i for i,(left,right) in enumerate(ranges) if left < end and right > start]

        def occurrences(needle: str) -> list[tuple[int,int]]:
            hits: list[tuple[int,int]] = []
            offset = 0
            while needle and (found := normalized_text.find(needle, offset)) >= 0:
                hits.append((found, found+len(needle)))
                offset = found+1
            return hits

        for person in self.config.participants:
            for alias in person.aliases:
                for start, end in occurrences(normalize(alias)):
                    indexes = affected(start,end)
                    sensitive_indexes.update(indexes)
                    if indexes:
                        rects = [boxes[i].rect for i in indexes]
                        pad = self.config.padding
                        rect = (max(0,min(r[0] for r in rects)-pad), max(0,min(r[1] for r in rects)-pad),
                                min(width,max(r[2] for r in rects)+pad), min(height,max(r[3] for r in rects)+pad))
                        values = name_groups.setdefault(rect,[])
                        if person.replacement not in values:
                            values.append(person.replacement)
        for literal in self.config.sensitive_literals:
            for start, end in occurrences(normalize(literal)):
                sensitive_indexes.update(affected(start,end))
        for pattern in (EMAIL,TW_ID):
            for match in pattern.finditer(normalized_text):
                sensitive_indexes.update(affected(match.start(),match.end()))
        cursor = 0
        for index,box in enumerate(boxes):
            end = cursor+len(box.text)
            compact = normalize(box.text)
            labels = {label for start, stop, label in spans if start < end and stop > cursor}
            literals = any(normalize(value) in compact for value in self.config.sensitive_literals)
            rules = any(pattern.search(box.text) for pattern in (EMAIL, TW_ID, PHONE, ADDRESS))
            # OCR sometimes inserts spaces into identifiers; check the compact form too.
            rules = rules or bool(EMAIL.search(compact) or TW_ID.search(compact))
            uncertain = box.confidence < self.config.low_confidence
            cue = CUES.search(box.text)
            if index in sensitive_indexes or literals or rules or labels or uncertain or cue:
                reason = 'known_or_cross_box' if index in sensitive_indexes else 'literal' if literals else 'rule' if rules else 'field' if cue else 'low_confidence' if uncertain else 'model'
                x1, y1, x2, y2 = box.rect
                pad = self.config.padding
                rect = (max(0, x1-pad), max(0, y1-pad), min(width, x2+pad), min(height, y2+pad))
                result.append(Mask(rect, reason))
                if cue:
                    # Cover separate value boxes to the right and just below a field label.
                    bottom = min(height, y2+max(16, 2*(y2-y1)))
                    result.append(Mask((max(0, x1-pad), max(0, y1-pad), width, bottom), 'field_guard'))
            cursor = end+1
        result.extend(Mask(rect,'participant',' / '.join(values)) for rect,values in name_groups.items())
        return result
