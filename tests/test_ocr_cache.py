from __future__ import annotations

import numpy as np
import pytest

from video_privacy.ocr_cache import ExactCropCache


def arguments(pixels: np.ndarray,point: int=10,decoder: str='greedy') -> tuple:
    return ('abc',32,100,None,None,[([point,point+20],pixels)],'',decoder,5,1,.1,.5,.003,0,'cpu')


class Recognizer:
    def __init__(self) -> None:self.calls=0
    def __call__(self,*args,**kwargs):
        self.calls+=1
        return [(args[5][0][0],str(int(args[5][0][1].sum())),.98)]


def test_identical_pixels_reuse_text_but_return_current_geometry() -> None:
    original=Recognizer();cache=ExactCropCache(original)
    image=np.zeros((10,20),dtype=np.uint8)
    first=cache(*arguments(image));second=cache(*arguments(image.copy(),point=200))
    assert original.calls==1 and cache.hits==1
    assert first[0][1:]==second[0][1:] and second[0][0]==[200,220]


def test_one_pixel_character_or_notification_change_never_reuses_old_result() -> None:
    original=Recognizer();cache=ExactCropCache(original)
    image=np.zeros((10,20),dtype=np.uint8)
    before=cache(*arguments(image));image[4,5]=1;after=cache(*arguments(image))
    assert original.calls==2 and before[0][1]!=after[0][1]


@pytest.mark.parametrize('change',['shape','dtype','decoder'])
def test_different_inputs_or_parameters_do_not_collide(change: str) -> None:
    original=Recognizer();cache=ExactCropCache(original)
    image=np.zeros((10,20),dtype=np.uint8);cache(*arguments(image))
    other=image.reshape(20,10) if change=='shape' else image.astype(np.int16) if change=='dtype' else image
    cache(*arguments(other,decoder='beamsearch' if change=='decoder' else 'greedy'))
    assert original.calls==2


def test_cache_is_bounded_and_evicted_entry_is_recomputed() -> None:
    original=Recognizer();cache=ExactCropCache(original,capacity=2)
    for value in [1,2,3,1]:cache(*arguments(np.full((2,3),value,dtype=np.uint8)))
    assert len(cache.cache)==2 and original.calls==4


def test_batch_and_unknown_keyword_calls_fall_back_to_original() -> None:
    original=Recognizer();cache=ExactCropCache(original)
    args=list(arguments(np.zeros((2,3),dtype=np.uint8)))
    args[5]=args[5]*2
    cache(*args);cache(*args)
    args=list(arguments(np.zeros((2,3),dtype=np.uint8)))
    cache(*args,custom=True)
    assert original.calls==3 and not cache.cache


def test_recognition_models_do_not_share_entries() -> None:
    original=Recognizer();cache=ExactCropCache(original)
    first=list(arguments(np.zeros((2,3),dtype=np.uint8)))
    second=list(first);second[3]=object()
    cache(*first);cache(*second)
    assert original.calls==2


def test_device_selection_uses_metal_only_when_available() -> None:
    from video_privacy.ocr import choose_device
    assert choose_device('auto',True)=='mps'
    assert choose_device('auto',False)=='cpu'
    assert choose_device('cpu',True)=='cpu'
    assert choose_device('mps',False)=='cpu'
    with pytest.raises(ValueError):choose_device('unknown',True)


@pytest.mark.parametrize('fail_at',['load','frame'])
def test_gpu_failure_retries_same_frame_on_cpu(monkeypatch,tmp_path,fail_at: str) -> None:
    import sys
    from types import ModuleType,SimpleNamespace
    from video_privacy.ocr import _worker
    events=[];received=[];clears=[]
    image=np.zeros((20,30,3),dtype=np.uint8)
    class Reader:
        def __init__(self,languages,gpu=False,**kwargs):
            self.gpu=gpu
            if gpu and fail_at=='load':raise RuntimeError('fixture-private-error')
        def detect(self,*args,**kwargs):
            return None
        def recognize(self,*args,**kwargs):
            return None
        def readtext(self,rgb,**kwargs):
            assert kwargs['min_size']==0
            self.detect(rgb)
            self.recognize(rgb)
            received.append((self.gpu,rgb))
            if self.gpu and fail_at=='frame':raise RuntimeError('fixture-private-error')
            return [([(1,1),(15,1),(15,10),(1,10)],'fixture',.99)]
    class Pipe:
        def __init__(self):self.inputs=iter([image,None])
        def send(self,value):events.append(value)
        def recv(self):return next(self.inputs)
        def close(self):pass
    engine=ModuleType('easyocr.easyocr');engine.get_text=lambda *a,**k:[]
    package=ModuleType('easyocr');package.__path__=[];package.Reader=Reader;package.easyocr=engine
    torch=ModuleType('torch');torch.backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda:True))
    torch.mps=SimpleNamespace(empty_cache=lambda:clears.append(True))
    monkeypatch.setitem(sys.modules,'easyocr',package)
    monkeypatch.setitem(sys.modules,'easyocr.easyocr',engine)
    monkeypatch.setitem(sys.modules,'torch',torch)
    monkeypatch.setenv('VPF_OCR_DEVICE','auto')
    _worker(Pipe(),str(tmp_path),('ch_tra','en'))
    assert [event[0] for event in events]==['ready','result']
    boxes,stats=events[-1][1]
    assert len(boxes)==1 and stats['ocr_device']=='cpu' and stats['ocr_cpu_fallbacks']==1
    assert received[-1][0] is False and received[-1][1] is image
    if fail_at=='frame':assert received[0][1] is image and clears
