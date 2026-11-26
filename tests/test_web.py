from __future__ import annotations

from contextlib import contextmanager
from http.cookiejar import CookieJar
import json
from pathlib import Path
import shutil
import threading
import time
from urllib.error import HTTPError
from urllib.request import HTTPCookieProcessor,Request,build_opener

import pytest

from video_privacy.web import Application,LocalServer,MAX_UPLOAD_BYTES
from test_privacy import create_video


def copy_processor(source: Path,destination: Path,*args: object,progress: object=None) -> dict[str,object]:
    progress(1)
    shutil.copyfile(source,destination)
    progress(3)
    return {'frames_processed':3}


@contextmanager
def running(tmp_path: Path,processor: object=copy_processor):
    app=Application(tmp_path/'models',processor=processor,prepare=lambda:None)
    server=LocalServer(('127.0.0.1',0),app)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    client=build_opener(HTTPCookieProcessor(CookieJar()))
    try:
        yield app,base,client
    finally:
        server.shutdown();server.server_close();thread.join(timeout=2);app.close()


def wait_done(app: Application) -> None:
    deadline=time.monotonic()+5
    while app.snapshot()['phase'] not in {'done','error'}:
        if time.monotonic()>deadline:raise AssertionError('job did not finish')
        time.sleep(.01)
    if app.worker is not None:app.worker.join(timeout=1)


def test_page_has_single_file_input_and_no_manual_config_form(tmp_path: Path) -> None:
    with running(tmp_path) as (app,base,client):
        with client.open(base) as response:
            html=response.read().decode()
            assert html.count('type="file"')==1
            assert '開始自動處理' in html and '下載處理後影片' in html
            assert '<textarea' not in html and '<select' not in html
            assert 'SameSite=Strict' in response.headers['Set-Cookie']
            assert "frame-ancestors 'none'" in response.headers['Content-Security-Policy']
        with client.open(base+'/api/state') as response:assert json.load(response)['phase']=='idle'


def test_real_upload_progress_preview_download_and_original_cleanup(tmp_path: Path) -> None:
    source=tmp_path/'synthetic.mp4';create_video(source)
    body=source.read_bytes()
    with running(tmp_path) as (app,base,client):
        client.open(base).close()
        request=Request(base+'/api/upload',data=body,headers={'X-Local-Token':app.token,'X-Video-Extension':'.mp4'})
        with client.open(request) as response:assert response.status==202
        wait_done(app);assert app.snapshot()['phase']=='done'
        assert app.snapshot()['frames']==3
        assert not list(app.folder.glob('input*'))
        with client.open(base+'/api/download') as response:
            assert response.read()==body
            assert 'attachment' in response.headers['Content-Disposition']
        with client.open(Request(base+'/api/video',headers={'Range':'bytes=0-15'})) as response:
            assert response.status==206 and response.read()==body[:16]
        with client.open(Request(base+'/api/video',method='HEAD')) as response:
            assert int(response.headers['Content-Length'])==len(body)
        with pytest.raises(HTTPError) as error:
            client.open(Request(base+'/api/video',headers={'Range':'bytes=999999999-'}))
        assert error.value.code==416


@pytest.mark.parametrize('mode',['no_cookie','bad_token','cross_origin','bad_host'])
def test_other_pages_cannot_start_a_local_upload(tmp_path: Path,mode: str) -> None:
    with running(tmp_path) as (app,base,client):
        if mode!='no_cookie':client.open(base).close()
        headers={'X-Local-Token':app.token,'X-Video-Extension':'.mp4'}
        if mode=='bad_token':headers['X-Local-Token']='invalid'
        if mode=='cross_origin':headers['Origin']='https://example.invalid'
        if mode=='bad_host':headers['Host']='example.invalid'
        with pytest.raises(HTTPError) as error:client.open(Request(base+'/api/upload',data=b'fictional',headers=headers))
        assert error.value.code==403
        assert app.snapshot()['phase']=='idle' and app.folder is None


def test_size_limit_and_download_before_completion(tmp_path: Path) -> None:
    with running(tmp_path) as (app,base,client):
        client.open(base).close()
        with pytest.raises(HTTPError) as error:
            client.open(Request(base+'/api/upload',data=b'x',headers={'X-Local-Token':app.token,'Content-Length':str(MAX_UPLOAD_BYTES+1)}))
        assert error.value.code==413
        with pytest.raises(HTTPError) as error:client.open(base+'/api/download')
        assert error.value.code==404


def test_failure_does_not_expose_exception_or_keep_original(tmp_path: Path) -> None:
    def broken(*args: object,**kwargs: object) -> dict[str,object]:
        raise ValueError('private-marker@example.invalid')
    source=tmp_path/'synthetic.mp4';create_video(source)
    with running(tmp_path,broken) as (app,base,client):
        client.open(base).close()
        client.open(Request(base+'/api/upload',data=source.read_bytes(),headers={'X-Local-Token':app.token})).close()
        wait_done(app)
        assert app.snapshot()['phase']=='error'
        assert 'private-marker' not in json.dumps(app.snapshot())
        assert not list(app.folder.glob('input*'))


def test_disk_entry_has_visible_startup_guide_and_relative_assets(tmp_path: Path) -> None:
    with running(tmp_path) as (app,base,client):
        with client.open(base) as response:
            html=response.read().decode()
        assert 'id="application" hidden' in html
        assert 'id="startupNotice"' in html and 'HTML 檔不能單獨處理影片' in html
        assert 'href="simple.css"' in html and 'src="simple.js"' in html
        with client.open(base+'/simple.js') as response:
            script=response.read().decode()
            assert "if(location.protocol==='http:'" in script
        with client.open(base+'/simple.css') as response:
            assert response.status==200 and '.startup-notice' in response.read().decode()



def test_launching_again_reuses_verified_same_version_service(tmp_path: Path,monkeypatch) -> None:
    from video_privacy.web import serve
    opened=[]
    monkeypatch.setattr('video_privacy.web.webbrowser.open',lambda url:opened.append(url))
    with running(tmp_path) as (app,base,client):
        port=int(base.rsplit(':',1)[1])
        serve(tmp_path/'models',port=port,open_browser=True)
        assert opened==[base+'/']
        assert app.snapshot()['phase']=='idle'


def test_port_conflict_never_opens_an_unverified_service(tmp_path: Path,monkeypatch) -> None:
    from video_privacy.web import serve
    opened=[]
    class Unverified:
        headers={}
        def __enter__(self):return self
        def __exit__(self,*args):pass
    monkeypatch.setattr('video_privacy.web.webbrowser.open',lambda url:opened.append(url))
    monkeypatch.setattr('video_privacy.web.urlopen',lambda *a,**k:Unverified())
    with running(tmp_path) as (app,base,client):
        with pytest.raises(ValueError,match='已被占用'):
            serve(tmp_path/'models',port=int(base.rsplit(':',1)[1]),open_browser=True)
        assert not opened
