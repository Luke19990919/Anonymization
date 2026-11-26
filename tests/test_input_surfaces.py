from __future__ import annotations

import numpy as np
import pytest
from PIL import Image,ImageDraw,ImageOps

from video_privacy.automatic import AutomaticDetector,input_surface,surface_ink
from video_privacy.detection import Box
from video_privacy.video import load_font


class NoSpans:
    def spans(self,text: str) -> list[tuple[int,int,str]]:
        return []


def screen(value: str='',caret: bool=False,dark: bool=False,offset: int=0) -> tuple[np.ndarray,Box,tuple[int,int,int,int] | None]:
    image=Image.new('RGB',(900,420),'white');draw=ImageDraw.Draw(image)
    label=Box((40,111+offset,151,148+offset),'登入密碼',.99)
    draw.rounded_rectangle((210,94+offset,850,164+offset),radius=7,fill='#fbfcff',outline='#b7c5d9',width=2)
    font=load_font(27);xy=(230,109+offset)
    target=None
    if value:
        draw.text(xy,value,font=font,fill='#1d304d')
        target=draw.textbbox(xy,value,font=font)
    if caret:
        x=230+int(draw.textlength(value,font=font))
        draw.line((x,110+offset,x,151+offset),fill='#2564cf',width=2)
    if dark:image=ImageOps.invert(image)
    return np.array(image),label,target


@pytest.mark.parametrize('dark',[False,True])
@pytest.mark.parametrize('offset',[0,90])
@pytest.mark.parametrize('caret',[False,True])
def test_empty_bounded_input_has_no_mask(dark: bool,offset: int,caret: bool) -> None:
    rgb,label,_=screen(caret=caret,dark=dark,offset=offset)
    surface=input_surface(rgb,label)
    assert surface is not None
    assert surface_ink(rgb,surface) is None
    detector=AutomaticDetector(NoSpans());detector.set_frame(rgb)
    # An empty OCR phantom on the far right must never be selected as the value.
    assert not detector.masks([label,Box((750,90+offset,840,190+offset),'',0)],900,420)


@pytest.mark.parametrize('value',['I','l','1','d','!','@','#','Demo09!@#','***'])
@pytest.mark.parametrize('dark',[False,True])
def test_content_is_protected_even_when_ocr_misses_every_character(value: str,dark: bool) -> None:
    rgb,label,target=screen(value=value,caret=True,dark=dark)
    detector=AutomaticDetector(NoSpans());detector.set_frame(rgb)
    masks=detector.masks([label],900,420)
    assert len(masks)==1 and masks[0].reason=='auto_password_field'
    x1,y1,x2,y2=masks[0].rect
    assert x1<=target[0] and y1<=target[1] and x2>=target[2] and y2>=target[3]
    assert 210<x1<x2<850  # protect content without a whole-row guard


def test_clearing_a_field_removes_prior_dynamic_mask() -> None:
    detector=AutomaticDetector(NoSpans())
    rgb,label,_=screen(value='Demo09!@#');detector.set_frame(rgb)
    assert detector.masks([label],900,420)
    rgb,label,_=screen(caret=True);detector.set_frame(rgb)
    assert detector.masks([label],900,420)==[]


def test_borderless_unknown_field_retains_conservative_fallback() -> None:
    rgb=np.full((420,900,3),255,dtype=np.uint8)
    detector=AutomaticDetector(NoSpans());detector.set_frame(rgb)
    masks=detector.masks([Box((40,110,150,145),'Password',.99)],900,420)
    assert masks and masks[0].rect[2]>750


@pytest.mark.parametrize('value',['I','l','1'])
def test_thin_dark_first_character_is_not_discarded_as_a_caret(value: str) -> None:
    image=Image.new('RGB',(300,60),'#fbfcff');draw=ImageDraw.Draw(image)
    draw.text((15,10),value,font=load_font(22),fill='#202020')
    assert surface_ink(np.array(image),(0,0,300,45)) is not None
