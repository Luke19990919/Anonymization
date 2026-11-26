from __future__ import annotations

import numpy as np
from PIL import Image,ImageDraw,ImageOps
import pytest

from video_privacy.automatic import AutomaticDetector,input_surface
from video_privacy.detection import Box
from video_privacy.video import load_font


class NoSpans:
    def spans(self,text: str) -> list[tuple[int,int,str]]:
        return []


def stacked(label_text: str,value: str,dark: bool=False,offset: int=0):
    image=Image.new('RGB',(900,600),'#fafafa');draw=ImageDraw.Draw(image)
    label=Box((230,90+offset,390,116+offset),label_text,.99)
    draw.text((230,90+offset),label_text,font=load_font(22),fill='#111111')
    draw.rectangle((230,132+offset,730,190+offset),fill='#ffffff',outline='#888888',width=2)
    target=None
    if value:
        draw.text((245,145+offset),value,font=load_font(24),fill='#111111')
        target=draw.textbbox((245,145+offset),value,font=load_font(24))
    if dark:image=ImageOps.invert(image)
    return np.array(image),label,target


@pytest.mark.parametrize('label_text',['手機號碼 *','電子信箱 *','Email','Mobile number','姓名','住址','身分證字號'])
@pytest.mark.parametrize('value',['1','a','王'])
@pytest.mark.parametrize('dark',[False,True])
def test_first_glyph_is_masked_without_value_ocr(label_text: str,value: str,dark: bool) -> None:
    rgb,label,target=stacked(label_text,value,dark)
    detector=AutomaticDetector(NoSpans());detector.set_frame(rgb)
    masks=detector.masks([label],900,600)
    assert target is not None
    assert any(m.rect[0]<=target[0] and m.rect[1]<=target[1] and
               m.rect[2]>=target[2] and m.rect[3]>=target[3] for m in masks)


@pytest.mark.parametrize('label_text',['手機號碼','電子信箱','姓名'])
def test_empty_stacked_field_is_not_masked(label_text: str) -> None:
    rgb,label,_=stacked(label_text,'')
    detector=AutomaticDetector(NoSpans());detector.set_frame(rgb)
    assert detector.masks([label],900,600)==[]


def test_stacked_surface_follows_layout_movement() -> None:
    for offset in (0,110,230):
        rgb,label,target=stacked('手機號碼','1',offset=offset)
        surface=input_surface(rgb,label)
        assert surface and target and surface[1]<=target[1]<target[3]<=surface[3]


def test_fully_protected_value_does_not_enter_nlp() -> None:
    class RecordingClassifier:
        def __init__(self):self.seen=[]
        def spans(self,text):self.seen.append(text);return []
    rgb,label,target=stacked('電子信箱','a')
    classifier=RecordingClassifier();detector=AutomaticDetector(classifier);detector.set_frame(rgb)
    assert target is not None
    value=Box(target,'a',1)
    masks=detector.masks([label,value],900,600)
    assert masks and all('\na' not in text for text in classifier.seen)
    assert detector.statistics['fully_protected_boxes_skipped_nlp']==1
