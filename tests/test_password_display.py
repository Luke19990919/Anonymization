from __future__ import annotations
import numpy as np
import pytest
from PIL import Image,ImageDraw
from video_privacy.automatic import AutomaticDetector,input_surface,password_looks_masked
from video_privacy.detection import Box
from video_privacy.video import load_font
from test_input_surfaces import screen,NoSpans


@pytest.mark.parametrize('dark',[False,True])
@pytest.mark.parametrize('caret',[False,True])
def test_password_bullets_are_preserved_even_if_ocr_hallucinates_text(dark: bool,caret: bool) -> None:
    rgb,label,_=screen(value='••••••',dark=dark,caret=caret)
    surface=input_surface(rgb,label)
    assert surface and password_looks_masked(rgb,surface)
    class HallucinatedPerson:
        def spans(self,text: str):
            start=text.find('Avery Demo')
            return [(start,start+10,'private_person')] if start>=0 else []
    detector=AutomaticDetector(HallucinatedPerson());detector.set_frame(rgb)
    assert detector.masks([label,Box((225,111,420,148),'Avery Demo',.8)],900,420)==[]


@pytest.mark.parametrize('value',['I','l','1','a','••a','••••!','....','***','oooo','0000','•','••'])
def test_plaintext_and_uncertain_patterns_still_mask(value: str) -> None:
    rgb,label,_=screen(value=value)
    detector=AutomaticDetector(NoSpans());detector.set_frame(rgb)
    assert any(m.reason=='auto_password_field' for m in detector.masks([label],900,420))


@pytest.mark.parametrize('hint',['隱藏','隱藏密碼','Hide','Hide password'])
def test_explicit_plaintext_display_state_overrides_bullet_shape(hint: str) -> None:
    rgb,label,_=screen(value='••••••')
    detector=AutomaticDetector(NoSpans());detector.set_frame(rgb)
    assert detector.masks([label,Box((855,111,895,148),hint,.99)],900,420)


def test_preservation_applies_to_password_not_username() -> None:
    rgb,label,_=screen(value='••••••')
    detector=AutomaticDetector(NoSpans());detector.set_frame(rgb)
    assert any(m.reason=='auto_account_field' for m in detector.masks([Box(label.rect,'Username',.99)],900,420))


def test_switching_plaintext_and_bullets_has_no_stale_mask() -> None:
    detector=AutomaticDetector(NoSpans())
    for value,expected in [('••••••',False),('Demo01!',True),('••••••',False)]:
        rgb,label,_=screen(value=value);detector.set_frame(rgb)
        assert bool(detector.masks([label],900,420)) is expected


def test_uneven_or_hollow_shapes_are_not_accepted_as_masking_bullets() -> None:
    rgb,label,_=screen()
    image=Image.fromarray(rgb);draw=ImageDraw.Draw(image)
    for x in [230,260,315]:draw.ellipse((x,125,x+7,132),fill='#223344')
    rgb=np.array(image);surface=input_surface(rgb,label)
    assert surface and not password_looks_masked(rgb,surface)
    draw=ImageDraw.Draw(image)
    draw.rectangle((215,100,845,158),fill='#fbfcff')
    for x in [230,260,290]:draw.ellipse((x,125,x+8,133),outline='#223344',width=1)
    rgb=np.array(image);assert not password_looks_masked(rgb,surface)
