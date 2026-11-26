"""Automatic OCR coordinate selection; no filenames, sidecars or known-person lists."""
from __future__ import annotations

from collections import Counter
import re

import numpy as np
from scipy import ndimage

from .config import Config, letter_id, normalize
from .field_surfaces import below_label_surface
from .search_privacy import SearchContext,inside,suggestion_surface,aligned_suggestion_region
from .detection import ADDRESS, EMAIL, TW_ID, Box, Detector, Mask, TextClassifier

FIELD_LABELS = {
    'name':{'姓名','受測者','受試者','使用者姓名','名字','name','fullname','participant'},
    'account':{'帳號','帳戶','登入帳號','使用者帳號','username','user','account','email','emailaddress','電子郵件地址','電子郵件','電子信箱','聯絡信箱','信箱','電子郵箱','郵箱','e-mail'},
    'phone':{'手機號碼','手機號','手機','電話','電話號碼','行動電話','行動電話號碼','phonenumber','mobilenumber','mobile','phone','telephone','tel'},
    'password':{'密碼','登入密碼','確認密碼','password','passcode','pwd'},
    'national_id':{'身分證','身份證','身分證字號','身份證字號','證號','識別碼','nationalid','idnumber'},
    'address':{'住址','地址','聯絡地址','聯絡位置','address','homeaddress'},
}
NOTIFICATION_APP = re.compile(r'^(?:line|signal|discord|messages|imessage|slack|wechat|微信)(?:\b|$)',re.I)
NOTIFICATION = re.compile(r'通知|私人訊息|notification|messenger|whatsapp|telegram|通訊訊息',re.I)
SURNAMES = set('陳林黃張李王吳劉蔡楊許鄭謝郭洪曾邱廖賴徐周葉蘇莊呂江何蕭羅高潘簡朱鍾彭游詹胡施沈余趙盧梁顏孫馬章范杜傅董姚石白任段雷韋廖温溫朱陈黄张刘杨许郑谢赖苏庄吕萧罗钟颜孙赵卢')
SAFE_SHORT = {'任務','姓名','受測者','受試者','確認','完成','設定','帳號','密碼','地址','操作','內容','資料','回饋','登入','聯絡','公開','範例','測試','結果','儲存','顯示','上傳','下載'}


def padded(rect: tuple[int,int,int,int],width: int,height: int,pad: int=10) -> tuple[int,int,int,int]:
    return max(0,rect[0]-pad),max(0,rect[1]-pad),min(width,rect[2]+pad),min(height,rect[3]+pad)


def field_kind(value: str) -> str | None:
    word=normalize(value).strip(':：*＊·.。')
    for kind,labels in FIELD_LABELS.items():
        if word in labels:return kind
    return None


def notification_regions(rgb: np.ndarray,boxes: list[Box]) -> list[Mask]:
    height,width=rgb.shape[:2]
    result: list[Mask]=[]
    for box in boxes:
        explicit=bool(NOTIFICATION.search(box.text))
        if not explicit and not NOTIFICATION_APP.search(box.text.strip()):continue
        x1,y1,x2,y2=box.rect
        # Window headings are not message cards.
        if y2<height*.12 and x1<width*.25:continue
        patch=rgb[max(0,y1-5):min(height,y2+5),max(0,x1-5):min(width,x2+5)]
        if not patch.size:continue
        bins=patch.reshape(-1,3)//16
        codes=bins[:,0].astype(int)*256+bins[:,1].astype(int)*16+bins[:,2]
        winner=int(np.bincount(codes,minlength=4096).argmax())
        color=np.median(patch.reshape(-1,3)[codes==winner],axis=0)
        similar=np.max(np.abs(rgb.astype(np.int16)-color.astype(np.int16)),axis=2)<=18
        labels,_=ndimage.label(similar)
        near=labels[max(0,y1-5):min(height,y2+5),max(0,x1-5):min(width,x2+5)]
        ids,counts=np.unique(near[near>0],return_counts=True)
        candidate=None
        if len(ids):
            selected=int(ids[int(counts.argmax())])
            slices=ndimage.find_objects(labels,max_label=selected)[selected-1]
            if slices:
                ys,xs=slices
                w,h=xs.stop-xs.start,ys.stop-ys.start
                if max(100,x2-x1)<=w<width*.72 and max(45,(y2-y1)*1.6)<=h<height*.55:
                    candidate=padded((xs.start,ys.start,xs.stop,ys.stop),width,height,10)
        if candidate is None and explicit:
            # Geometry fallback for corner notifications whose background is not a flat card.
            if x1>width*.6:
                candidate=(max(0,x1-30),max(0,y1-25),width,min(height,y1+int(height*.24)))
            elif x1<width*.4 and y1>height*.13:
                candidate=(max(0,x1-25),max(0,y1-25),min(width,x1+int(width*.38)),min(height,y1+int(height*.24)))
        # A guessed card must contain its own detected heading; reject inconsistent
        # fallbacks such as a long explanatory footer containing the word notification.
        if candidate is not None and not (candidate[0]<=x1 and candidate[1]<=y1 and candidate[2]>=x2 and candidate[3]>=y2):
            candidate=None
        if candidate is not None and candidate not in [m.rect for m in result]:
            result.append(Mask(candidate,'auto_notification'))
    return result


def input_surface(rgb: np.ndarray,label: Box) -> tuple[int,int,int,int] | None:
    """Find a closed, flat input surface beside a label, without fixed coordinates."""
    height,width=rgb.shape[:2]
    x1,y1,x2,y2=label.rect
    row=max(0,min(height-1,(y1+y2)//2))
    left=max(0,min(width,x2+8))
    samples=rgb[row,left:]
    if not samples.size:return None
    quantized=samples.astype(np.int32)//16
    codes=quantized[:,0]*256+quantized[:,1]*16+quantized[:,2]
    unique,counts=np.unique(codes,return_counts=True)
    max_height=min(height*.2,(y2-y1)*3)
    search_top=max(0,row-int(max_height)-2)
    search_bottom=min(height,row+int(max_height)+3)
    search=rgb[search_top:search_bottom]
    candidates=[]
    for code in unique[np.argsort(counts)[-3:]]:
        color=np.median(samples[codes==code],axis=0)
        similar=np.max(np.abs(search.astype(np.int16)-color.astype(np.int16)),axis=2)<=14
        components,_=ndimage.label(similar)
        ids=np.unique(components[row-search_top,left:]);ids=ids[ids>0]
        if not len(ids):continue
        bounds=ndimage.find_objects(components)
        for identity in ids:
            region=bounds[int(identity)-1]
            if region is None:continue
            ys,xs=region;w=xs.stop-xs.start;h=ys.stop-ys.start
            # Components connected beyond the band cannot be a bounded eligible input.
            if (ys.start==0 and search_top>0) or (ys.stop==len(search) and search_bottom<height):continue
            if (xs.start>x2+3 and max(100,(x2-x1)*1.1)<=w<width*.9
                    and max(24,(y2-y1)*.65)<=h<=min(height*.2,(y2-y1)*3)
                    and ys.start<=row-search_top<ys.stop):
                patch=components[ys,xs]
                if np.mean(patch==identity)>.7:
                    candidates.append((xs.start,ys.start+search_top,xs.stop,ys.stop+search_top))
    return min(candidates,key=lambda r:(r[0],(r[2]-r[0])*(r[3]-r[1]))) if candidates else below_label_surface(rgb,label)


def _surface_foreground(rgb: np.ndarray,rect: tuple[int,int,int,int]) -> tuple[int,int,np.ndarray] | None:
    """Foreground within a bounded input, excluding borders and clear colored carets."""
    x1,y1,x2,y2=rect
    inset=4
    left,top,right,bottom=x1+inset,y1+inset,x2-inset,y2-inset
    patch=rgb[top:bottom,left:right]
    if not patch.size:return None
    background=np.median(patch.reshape(-1,3),axis=0)
    ink=np.max(np.abs(patch.astype(np.int16)-background.astype(np.int16)),axis=2)>45
    labels,_=ndimage.label(ink)
    for identity,region in enumerate(ndimage.find_objects(labels),1):
        if region is None:continue
        ys,xs=region
        w,h=xs.stop-xs.start,ys.stop-ys.start
        component_color=np.median(patch[labels==identity],axis=0)
        caret_like=(w<=max(2,int((y2-y1)*.05)) and h>=(y2-y1)*.52
                    and h/max(1,w)>=12 and np.ptp(component_color)>60)
        if caret_like:
            ink[labels==identity]=False
    return left,top,ink


def surface_ink(rgb: np.ndarray,rect: tuple[int,int,int,int]) -> tuple[int,int,int,int] | None:
    foreground=_surface_foreground(rgb,rect)
    if foreground is None:return None
    left,top,ink=foreground
    x1,y1,x2,y2=rect
    ys,xs=np.nonzero(ink)
    if not len(xs):return None
    return (max(x1,left+int(xs.min())-18),max(y1,top+int(ys.min())-18),
            min(x2,left+int(xs.max())+19),min(y2,top+int(ys.max())+19))


def password_looks_masked(rgb: np.ndarray,rect: tuple[int,int,int,int]) -> bool:
    """Conservative evidence of >=3 aligned, equally spaced filled circular bullets.

    This is a visual style heuristic, not access to an HTML input's password state.
    Periods, stars, mixed text, hollow letters and uncertain patterns remain protected.
    """
    foreground=_surface_foreground(rgb,rect)
    if foreground is None:return False
    left,top,ink=foreground
    labels,_=ndimage.label(ink)
    bounds=ndimage.find_objects(labels)
    if not 3<=len(bounds)<=256:return False
    components=[]
    for identity,region in enumerate(bounds,1):
        if region is None:return False
        ys,xs=region;w,h=xs.stop-xs.start,ys.stop-ys.start
        glyph=labels[ys,xs]==identity
        if min(w,h)<3 or max(w,h)>(rect[3]-rect[1])*.25:return False
        if not .65<=w/h<=1.5:return False
        if not .60<=float(glyph.mean())<=.94:return False
        if np.any(ndimage.binary_fill_holes(glyph)&~glyph):return False
        corners=int(glyph[0,0])+int(glyph[0,-1])+int(glyph[-1,0])+int(glyph[-1,-1])
        if corners>1:return False
        components.append(((xs.start+xs.stop)/2,(ys.start+ys.stop)/2,w,h))
    components.sort()
    a=np.array(components);mw,mh=np.median(a[:,2:4],axis=0)
    if np.ptp(a[:,2])>max(2,mw*.25) or np.ptp(a[:,3])>max(2,mh*.25):return False
    if np.ptp(a[:,1])>max(2,mh*.25):return False
    vertical=(top+float(np.median(a[:,1]))-rect[1])/(rect[3]-rect[1])
    if not .30<=vertical<=.65:return False
    gaps=np.diff(a[:,0]);spacing=float(np.median(gaps))
    return bool(spacing>=mw*1.3 and np.max(np.abs(gaps-spacing))<=max(1,spacing*.12))


def password_visibility_hint(boxes: list[Box],surface: tuple[int,int,int,int]) -> bool:
    """A nearby 'hide password' action contradicts treating visible glyphs as masked."""
    x1,y1,x2,y2=surface
    return any(normalize(b.text).strip(':：') in {'hide','hidepassword','隱藏','隱藏密碼'}
               and (b.rect[0]+b.rect[2])/2>=x1 and y1<=(b.rect[1]+b.rect[3])/2<=y2
               for b in boxes)


class AutomaticDetector:
    def __init__(self,classifier: TextClassifier, *, experimental_search_context: bool=False) -> None:
        self.classifier=classifier
        self.notification_motion_guard_seconds=.75
        # Confidence measures OCR readability, not whether the text is private.
        # Keep rules/model/field protection; never redact solely for low confidence.
        self.base=Detector(Config((),(),(),low_confidence=0.0),classifier)
        self.names: dict[str,str]={}
        self.anchors: dict[str,str]={}
        self.next_person=0
        self.statistics: Counter[str]=Counter()
        self.frame_image: np.ndarray | None=None
        self.experimental_search_context=experimental_search_context
        self.search_context=SearchContext()
        self.public_search_page=False

    def set_frame(self,rgb: np.ndarray) -> None:
        self.frame_image=rgb

    def identity(self,name: str,anchors: list[str] | None=None) -> str:
        key=normalize(name)
        # IDs take precedence over OCR spelling variants; links stay in memory only.
        linked=next((self.anchors[a] for a in anchors or [] if a in self.anchors),None)
        value=linked or self.names.get(key)
        if value is None:
            value=f'受測者 {letter_id(self.next_person)}';self.next_person+=1
        if key:self.names[key]=value
        for anchor in anchors or []:self.anchors[anchor]=value
        return value

    def masks(self,boxes: list[Box],width: int,height: int) -> list[Mask]:
        boxes=sorted(boxes,key=lambda b:(b.rect[1],b.rect[0]))
        search_regions=[]
        if self.experimental_search_context and self.frame_image is not None:
            surface=self.search_context.locate(self.frame_image,boxes)
            if surface is not None:
                search_regions.append(surface)
                suggestions=suggestion_surface(self.frame_image,surface)
                if suggestions is None:
                    suggestions=aligned_suggestion_region(boxes,surface,height)
                if suggestions is not None:search_regions.append(suggestions)
                self.public_search_page=self.public_search_page or any(
                    'youtube.com' in b.text.lower() and b.rect[1]<height*.15 for b in boxes)
            else:
                self.public_search_page=False
        def in_search(box: Box) -> bool:
            return any(inside(box,region) for region in search_regions)
        notifications=notification_regions(self.frame_image,boxes) if self.frame_image is not None else []
        def in_notification(box: Box) -> bool:
            x=(box.rect[0]+box.rect[2])/2;y=(box.rect[1]+box.rect[3])/2
            return any(m.rect[0]<=x<=m.rect[2] and m.rect[1]<=y<=m.rect[3] for m in notifications)
        boxes=[b for b in boxes if not in_notification(b)]
        fields: list[tuple[str,Box,Box | None]]=[]
        for label in boxes:
            kind=field_kind(label.text)
            if kind is None:continue
            x1,y1,x2,y2=label.rect
            candidates=[]
            for value in boxes:
                if value is label or not value.text.strip() or field_kind(value.text):continue
                a,b,c,d=value.rect
                overlap=max(0,min(y2,d)-max(y1,b))
                if a>x2-8 and d-b<=(y2-y1)*2 and overlap>=min(y2-y1,d-b)*.35:
                    candidates.append(value)
            value=min(candidates,key=lambda b:b.rect[0]-x2) if candidates else None
            fields.append((kind,label,value))
        output=list(notifications)
        handled_names: set[tuple[int,int,int,int]]=set()
        masked_password_surfaces: list[tuple[int,int,int,int]]=[]
        pixel_protected_rects: list[tuple[int,int,int,int]]=[]
        for kind,label,value in fields:
            surface=input_surface(self.frame_image,label) if self.frame_image is not None and kind in {'account','phone','password','name','address','national_id'} else None
            if surface is not None:
                contained=[b for b in boxes if b is not label and b.text.strip() and inside(b,surface)]
                if contained:value=min(contained,key=lambda b:b.rect[0])
                rect=surface_ink(self.frame_image,surface)
                if rect is None:
                    self.statistics['confirmed_empty_inputs']+=1
                    continue
                if kind=='password' and not password_visibility_hint(boxes,surface) and password_looks_masked(self.frame_image,surface):
                    masked_password_surfaces.append(surface)
                    self.statistics['visually_masked_password_inputs_preserved']+=1
                    continue
                self.statistics['pixel_protected_inputs']+=1
                pixel_protected_rects.append(rect)
            elif value is not None:
                rect=padded(value.rect,width,height)
            else:
                x1,y1,x2,y2=label.rect
                rect=(min(width-1,x2+8),max(0,y1-12),min(width,max(x2+40,int(x1+width*.85))),min(height,y2+24))
                self.statistics['field_without_readable_value']+=1
            replacement=None
            if kind=='name':
                anchors=[]
                if value is not None:
                    for anchor_kind,anchor_label,anchor_value in fields:
                        if anchor_value is None or anchor_kind not in {'national_id','account'}:continue
                        if abs(anchor_value.rect[0]-value.rect[0])>width*.18:continue
                        if abs(anchor_value.rect[1]-value.rect[1])>height*.65:continue
                        raw=normalize(anchor_value.text)
                        if anchor_kind=='national_id':
                            raw=raw.upper().replace('O','0').replace('I','1')
                            anchors.insert(0,'id:'+raw)
                        elif EMAIL.search(raw):anchors.append('email:'+raw)
                replacement=self.identity(value.text,anchors) if value is not None else '受測者（待確認）'
                handled_names.add(value.rect if value else rect)
            output.append(Mask(rect,'auto_'+kind+'_field',replacement))
        if masked_password_surfaces:
            # Keep model/OCR hallucinations about bullet glyphs from reintroducing a mask.
            boxes=[b for b in boxes if not any(r[0]<=(b.rect[0]+b.rect[2])/2<=r[2]
                                              and r[1]<=(b.rect[1]+b.rect[3])/2<=r[3]
                                              for r in masked_password_surfaces)]
        # Full core coverage is already established; no recognizer is needed for
        # those boxes again. Partial overlap never qualifies for this shortcut.
        retained=[]
        for box in boxes:
            x1,y1,x2,y2=box.rect
            covered=any(a<=x1 and b<=y1 and c>=x2 and d>=y2 for a,b,c,d in pixel_protected_rects)
            if covered:
                self.statistics['fully_protected_boxes_skipped_nlp']+=1
            else:
                retained.append(box)
        boxes=retained
        # Model and format detections supplement field layout; broad legacy field guards are excluded.
        output.extend(m for m in self.base.masks(boxes,width,height) if m.reason not in {'field','field_guard'})
        joined='\n'.join(b.text for b in boxes)
        all_spans=self.classifier.spans(joined)
        person_spans=[(a,b) for a,b,label in all_spans if label=='private_person']
        cursor=0
        for box in boxes:
            compact=normalize(box.text)
            model_name=any(a<cursor+len(box.text) and b>cursor for a,b in person_spans)
            cjk_name=(2<=len(compact)<=4 and compact[0] in SURNAMES and compact not in SAFE_SHORT
                      and re.fullmatch(r'[\u3400-\u9fff]+',compact) is not None)
            contextual=in_search(box)
            # Search icons can be merged into text; short prefixes must remain protected.
            query=compact.lstrip('QqOo○◯⌕🔍•。')
            name_prefix=bool(query and query[0] in SURNAMES and re.match(r'[\u3400-\u9fff]',query))
            if contextual and name_prefix:
                output.append(Mask(padded(box.rect,width,height),'search_private_name'))
            allow_person=not self.public_search_page or contextual
            if allow_person and box.rect not in handled_names and (model_name or cjk_name) and field_kind(box.text) is None:
                output.append(Mask(padded(box.rect,width,height),'auto_person',self.identity(box.text)))
            if self.public_search_page and not contextual:
                labels={label for a,b,label in all_spans if a<cursor+len(box.text) and b>cursor}
                if labels=={'private_person'}:
                    rect=padded(box.rect,width,height,self.base.config.padding)
                    output=[m for m in output if not (m.reason=='model' and m.rect==rect)]
            cursor+=len(box.text)+1
        self.statistics['low_confidence_boxes_seen']+=sum(b.confidence<.3 for b in boxes)
        self.statistics['frames_analyzed']+=1
        return output

    def summary(self) -> dict[str,object]:
        return {'mode':'automatic','notifications':'automatic keyword/card geometry heuristics; review required',
                'inferred_participants':self.next_person,
                'automatic_statistics':dict(self.statistics),
                'automatic_limitations':['OCR/model detection can miss or overmask text',
                                        'notification detection is heuristic, not universal',
                                        'participant identity links require human verification'],
                'raw_name_mapping_saved':False,
                'experimental_search_context_enabled':self.experimental_search_context,
                'search_context_policy':'experimental pixel-validated search context; surname prefixes protected; recognized YouTube background person labels preserved; review required',
                'password_display_policy':'preserve visually confirmed repeated circular bullets in bounded password inputs; mask plaintext or uncertain states; visible hide-action overrides bullet heuristic',
                'password_bullet_length_remains_visible':True,
                'low_confidence_policy':'confidence alone does not trigger masking; review uncertain text'}
