from __future__ import annotations

import numpy as np
from PIL import Image,ImageDraw
import pytest

from video_privacy.automatic import AutomaticDetector
from video_privacy.detection import Box
from video_privacy.search_privacy import SearchContext,suggestion_surface


class NoSpans:
    def spans(self,text: str) -> list[tuple[int,int,str]]:
        return []


def page(query: str='搜尋',popup: bool=True,popup_gap: int=7):
    image=Image.new('RGB',(900,600),'#0f0f0f');draw=ImageDraw.Draw(image)
    draw.rounded_rectangle((220,70,720,118),radius=15,fill='#171717',outline='#444444')
    if popup:draw.rectangle((220,118+popup_gap,720,420),fill='#282828')
    return np.array(image),[Box((20,5,140,24),'youtube.com',1),Box((245,85,300,105),query,1)]


def test_surface_is_found_without_fixed_screen_coordinates() -> None:
    rgb,boxes=page()
    context=SearchContext();surface=context.locate(rgb,boxes)
    assert surface and surface[0]<245<surface[2] and surface[1]<85<surface[3]
    assert suggestion_surface(rgb,surface) is not None
    # A moved page invalidates prior geometry rather than masking stale coordinates.
    moved=np.roll(rgb,160,axis=0)
    assert context.locate(moved,[]) is None


@pytest.mark.parametrize('query',['王','王小','王小明','Q王小明跳舞'])
def test_typed_prefix_and_suggestion_name_are_covered(query: str) -> None:
    detector=AutomaticDetector(NoSpans(),experimental_search_context=True);rgb,boxes=page()
    detector.set_frame(rgb);detector.masks(boxes,900,600)
    rgb,boxes=page(query)
    boxes.append(Box((245,150,440,175),'Q王小明跳舞',.3))
    detector.set_frame(rgb);masks=detector.masks(boxes,900,600)
    for target in [boxes[1],boxes[2]]:
        assert any(m.rect[0]<=target.rect[0] and m.rect[1]<=target.rect[1] and
                   m.rect[2]>=target.rect[2] and m.rect[3]>=target.rect[3] for m in masks)


def test_general_search_is_clear_and_public_person_is_not_aliased() -> None:
    class PersonModel:
        def spans(self,text: str):
            name='Avery Example'
            start=text.find(name)
            return [(start,start+len(name),'private_person')] if start>=0 else []
    detector=AutomaticDetector(PersonModel(),experimental_search_context=True);rgb,boxes=page()
    detector.set_frame(rgb);detector.masks(boxes,900,600)
    rgb,boxes=page('python 教學')
    boxes.append(Box((480,500,680,530),'Avery Example',1))
    detector.set_frame(rgb)
    assert detector.masks(boxes,900,600)==[]


def test_private_identifier_search_remains_protected() -> None:
    detector=AutomaticDetector(NoSpans(),experimental_search_context=True);rgb,boxes=page()
    detector.set_frame(rgb);detector.masks(boxes,900,600)
    rgb,boxes=page('alex@example.invalid')
    detector.set_frame(rgb)
    assert detector.masks(boxes,900,600)


def test_popup_gap_does_not_drop_sensitive_suggestions() -> None:
    detector=AutomaticDetector(NoSpans(),experimental_search_context=True)
    rgb,boxes=page();detector.set_frame(rgb);detector.masks(boxes,900,600)
    rgb,boxes=page('python',popup_gap=31)
    target=Box((245,180,440,205),'Q王小明跳舞',.3)
    detector.set_frame(rgb)
    masks=detector.masks(boxes+[target],900,600)
    assert any(m.reason=='search_private_name' and m.rect[1]<=180 and m.rect[3]>=205 for m in masks)


def test_same_color_menu_uses_aligned_search_icons_not_page_colors() -> None:
    from video_privacy.search_privacy import aligned_suggestion_region
    field=(220,70,720,118)
    rows=[Box((240,y,440,y+22),text,1) for y,text in
          [(150,'Q python'),(200,'Q王小明跳舞'),(250,'Q how to')]]
    region=aligned_suggestion_region(rows,field,600)
    assert region and region[1]<=200 and region[3]>=222


def test_public_text_without_search_icons_is_not_a_suggestion_panel() -> None:
    from video_privacy.search_privacy import aligned_suggestion_region
    rows=[Box((240,y,440,y+22),'Public video title',1) for y in (150,200,250)]
    assert aligned_suggestion_region(rows,(220,70,720,118),600) is None


def test_widely_separated_thumbnail_rows_are_not_a_suggestion_panel() -> None:
    from video_privacy.search_privacy import aligned_suggestion_region
    rows=[Box((240,y,440,y+22),'Q fictional title',1) for y in (150,300,450)]
    assert aligned_suggestion_region(rows,(220,70,720,118),600) is None
