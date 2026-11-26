from __future__ import annotations

import numpy as np

from video_privacy.automatic import AutomaticDetector, notification_regions
from video_privacy.detection import Box


class NoSpans:
    def spans(self,text: str) -> list[tuple[int,int,str]]:
        return []


def form(name: str,identifier: str,email: str='person@example.invalid') -> list[Box]:
    return [Box((40,100,110,130),'受測者',.99),Box((200,100,350,135),name,.99),
            Box((40,190,110,220),'識別碼',.99),Box((200,190,390,225),identifier,.99),
            Box((40,280,130,310),'聯絡信箱',.99),Box((200,280,500,315),email,.99)]


def test_automatic_fields_need_no_manual_names_or_rectangles() -> None:
    detector=AutomaticDetector(NoSpans())
    masks=detector.masks(form('林範例','Z100000002'),800,600)
    name=next(m for m in masks if m.reason=='auto_name_field')
    assert name.replacement=='受測者 A'
    assert name.rect[0]<=200 and name.rect[2]>=350
    assert any(m.reason=='auto_national_id_field' for m in masks)
    assert any(m.reason=='auto_account_field' for m in masks)


def test_ids_link_chinese_english_aliases_without_a_roster() -> None:
    detector=AutomaticDetector(NoSpans())
    for name in ['林範例','Avery Demo','林範例']:
        masks=detector.masks(form(name,'Z100000002'),800,600)
        assert next(m for m in masks if m.reason=='auto_name_field').replacement=='受測者 A'


def test_multiple_people_return_to_the_same_alias() -> None:
    detector=AutomaticDetector(NoSpans())
    for name,identifier,email,expected in [('林範例','Z100000002','a@example.invalid','A'),('陳樣本','Z100000003','b@example.invalid','B'),('吳測試','Z100000004','c@example.invalid','C'),('林範例','Z100000002','a@example.invalid','A')]:
        masks=detector.masks(form(name,identifier,email),800,600)
        assert next(m for m in masks if m.reason=='auto_name_field').replacement==f'受測者 {expected}'


def test_password_without_readable_value_still_gets_a_guard() -> None:
    masks=AutomaticDetector(NoSpans()).masks([Box((30,100,130,130),'Password',.99)],800,600)
    assert any(m.reason=='auto_password_field' and m.rect[2]>600 for m in masks)


def test_cjk_name_is_conservatively_masked_without_model_detection() -> None:
    masks=AutomaticDetector(NoSpans()).masks([Box((50,50,140,90),'林範例',.95)],800,600)
    assert any(m.reason=='auto_person' and m.replacement=='受測者 A' for m in masks)


def test_normal_interface_labels_do_not_become_people() -> None:
    boxes=[Box((20,20,250,60),'一般操作流程',.99),Box((30,100,100,140),'任務',.99)]
    assert not AutomaticDetector(NoSpans()).masks(boxes,800,600)


def test_entire_notification_card_is_selected_from_pixels() -> None:
    rgb=np.full((600,800,3),255,dtype=np.uint8)
    rgb[120:270,410:780]=[107,142,194]
    rgb[122:268,412:778]=[220,234,255]
    boxes=[Box((425,130,720,160),'私人通訊通知',.99)]
    regions=notification_regions(rgb,boxes)
    assert regions and regions[0].rect[0]<=410 and regions[0].rect[2]>=780
    assert regions[0].rect[1]<=120 and regions[0].rect[3]>=270


def test_window_heading_is_not_a_notification() -> None:
    rgb=np.full((600,800,3),255,dtype=np.uint8)
    assert not notification_regions(rgb,[Box((20,10,260,40),'單畫面短暫通知',.99)])


def test_readability_uncertainty_alone_does_not_mask_normal_text() -> None:
    boxes=[Box((67,151,109,185),'一般',.12)]
    assert AutomaticDetector(NoSpans()).masks(boxes,1280,720)==[]


def test_uncertain_email_and_password_field_remain_protected() -> None:
    detector=AutomaticDetector(NoSpans())
    boxes=[Box((40,100,130,130),'Password',.15),
           Box((210,100,360,135),'demo-secret',.12),
           Box((50,240,400,275),'fictional@example.invalid',.1)]
    masks=detector.masks(boxes,800,600)
    assert any(m.reason=='auto_password_field' for m in masks)
    assert any(m.rect[0]<=50 and m.rect[2]>=400 and m.rect[1]<=240 for m in masks)


def test_uncertain_model_detected_person_is_still_protected() -> None:
    class PersonSpans:
        def spans(self,text: str) -> list[tuple[int,int,str]]:
            return [(0,len(text),'private_person')]
    masks=AutomaticDetector(PersonSpans()).masks([Box((50,100,250,135),'Avery Demo',.1)],800,600)
    assert any(m.reason=='auto_person' and m.replacement=='受測者 A' for m in masks)



def test_long_notification_explanation_is_not_a_card_fallback() -> None:
    rgb=np.full((720,1280,3),255,dtype=np.uint8)
    boxes=[Box((40,660,700,700),'通知中的姓名與訊息均為虛構，不含真實私人內容',.99)]
    assert notification_regions(rgb,boxes)==[]


def test_short_notification_heading_keeps_conservative_corner_fallback() -> None:
    rgb=np.full((720,1280,3),255,dtype=np.uint8)
    masks=notification_regions(rgb,[Box((30,130,240,165),'私人通訊通知',.99)])
    assert masks and masks[0].rect[0]<=30 and masks[0].rect[2]>=240


def test_branded_private_notification_does_not_require_pii_or_notification_word() -> None:
    rgb=np.full((600,800,3),255,dtype=np.uint8)
    rgb[120:270,410:780]=[107,142,194]
    rgb[122:268,412:778]=[220,234,255]
    for app in ('LINE','Signal','Discord'):
        regions=notification_regions(rgb,[Box((425,130,510,160),app,.99)])
        assert regions and regions[0].rect[0]<=410 and regions[0].rect[2]>=780
        assert regions[0].rect[1]<=120 and regions[0].rect[3]>=270


def test_brand_word_without_a_bounded_card_never_uses_corner_fallback() -> None:
    rgb=np.full((600,800,3),255,dtype=np.uint8)
    assert notification_regions(rgb,[Box((600,130,690,160),'LINE',.99)])==[]
