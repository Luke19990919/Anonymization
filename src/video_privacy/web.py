"""Loopback-only consumer UI. Video bytes never go to a hosted service."""
from __future__ import annotations

from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json
import errno
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Callable
from urllib.parse import urlsplit
from urllib.request import urlopen
import webbrowser

from . import __version__

MAX_UPLOAD_BYTES=2*1024*1024*1024
Processor=Callable[...,dict[str,object]]


class Application:
    def __init__(self,models: Path,checkpoint: Path | None=None,
                 processor: Processor | None=None,prepare: Callable[[],None] | None=None) -> None:
        self.models=models.resolve()
        self.checkpoint=checkpoint.resolve() if checkpoint else None
        self.processor=processor
        self.preparer=prepare
        self.token=secrets.token_urlsafe(32)
        self.workspace=Path(tempfile.mkdtemp(prefix='vpf-web-'))
        os.chmod(self.workspace,0o700)
        self.lock=threading.Lock()
        self.stopping=threading.Event()
        self.worker: threading.Thread | None=None
        self.folder: Path | None=None
        self.output: Path | None=None
        self.state: dict[str,object]={'phase':'idle','frames':0,'total_frames':0,'error':None}

    def snapshot(self) -> dict[str,object]:
        with self.lock:
            result=dict(self.state)
            if 'started_at' in result:
                result['elapsed_seconds']=max(0,float(result.get('finished_at',time.time()))-float(result['started_at']))
            return result

    def update(self,**changes: object) -> None:
        with self.lock:self.state.update(changes)

    def reserve(self,extension: str) -> Path:
        with self.lock:
            if self.state['phase'] in {'uploading','preparing','processing'}:
                raise RuntimeError('已有影片正在處理，請等待完成。')
            previous=self.folder
            self.folder=self.workspace/secrets.token_hex(12)
            self.folder.mkdir(mode=0o700)
            self.output=self.folder/'processed-video.mp4'
            self.state={'phase':'uploading','frames':0,'total_frames':0,'error':None,'started_at':time.time()}
        if previous is not None:shutil.rmtree(previous,ignore_errors=True)
        return self.folder/('input'+extension)

    def upload_failed(self,source: Path) -> None:
        source.unlink(missing_ok=True)
        self.update(phase='error',error='影片未完整傳入，請重新選擇後再試。')

    def launch(self,source: Path) -> None:
        self.worker=threading.Thread(target=self._process,args=(source,),daemon=True)
        self.worker.start()

    def _prepare(self) -> None:
        if self.preparer is not None:
            self.preparer();return
        from .models import check_models,checkpoint_path
        try:check_models(self.models,checkpoint_path(self.models,self.checkpoint))
        except ValueError:
            self.update(phase='preparing')
            # Preparation loads OCR libraries; keep them separate from the video process.
            command=[sys.executable,'-m','video_privacy.cli','prepare-models','--models',str(self.models)]
            if self.checkpoint is not None:command+=['--checkpoint',str(self.checkpoint)]
            environment=dict(os.environ);environment['HF_HUB_OFFLINE']='0'
            with subprocess.Popen(command,env=environment,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL) as task:
                while task.poll() is None:
                    if self.stopping.wait(.2):
                        task.terminate();task.wait(timeout=5)
                        raise RuntimeError('準備已停止。')
                if task.returncode!=0:raise RuntimeError('所需檔案準備失敗。')

    def _process(self,source: Path) -> None:
        try:
            from .video import inspect_video
            info=inspect_video(source)
            self.update(total_frames=info['declared_frames'])
            self._prepare()
            if self.stopping.is_set():return
            self.update(phase='processing')
            def progress(count: int) -> None:
                if self.stopping.is_set():raise RuntimeError('本機處理已停止。')
                self.update(frames=count)
            if self.processor is None:
                from .processing import run_automatic
                processor=run_automatic
            else:processor=self.processor
            assert self.output is not None
            report=processor(source,self.output,self.models,self.checkpoint,progress=progress)
            self.update(phase='done',finished_at=time.time(),frames=report['frames_processed'],
                        total_frames=report['frames_processed'],audio_removed=True,
                        review_required=True,mask_style='gaussian_blur')
        except Exception:
            phase=self.snapshot()['phase']
            message='所需檔案準備失敗，請確認網路連線後再試。' if phase=='preparing' else '影片處理未完成。請確認影片可正常播放，或先試用附帶的測試影片。'
            self.update(phase='error',error=message)
        finally:
            source.unlink(missing_ok=True)

    def close(self) -> None:
        self.stopping.set()
        if self.worker is not None:self.worker.join(timeout=5)
        shutil.rmtree(self.workspace,ignore_errors=True)


class LocalServer(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,address: tuple[str,int],application: Application) -> None:
        self.application=application
        super().__init__(address,Handler)


class Handler(BaseHTTPRequestHandler):
    server: LocalServer
    protocol_version='HTTP/1.1'

    def log_message(self,*args: object) -> None:
        pass

    def headers_for(self,status: int,content_type: str,length: int,extra: dict[str,str] | None=None) -> None:
        self.send_response(status)
        self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(length))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Video-Privacy-Server','local-vpf')
        self.send_header('X-Video-Privacy-Version',__version__)
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('X-Frame-Options','DENY')
        self.send_header('Referrer-Policy','no-referrer')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; media-src 'self' blob:; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'; base-uri 'none'")
        for name,value in (extra or {}).items():self.send_header(name,value)
        self.end_headers()

    def json(self,status: int,payload: dict[str,object]) -> None:
        body=json.dumps(payload,ensure_ascii=False).encode()
        self.headers_for(status,'application/json; charset=utf-8',len(body))
        self.wfile.write(body)

    def allowed_host(self) -> bool:
        host=self.headers.get('Host','')
        return host in {f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}

    def authorized(self) -> bool:
        cookie=SimpleCookie()
        try:cookie.load(self.headers.get('Cookie',''))
        except Exception:return False
        token=cookie.get('vpf_session')
        return self.allowed_host() and token is not None and secrets.compare_digest(token.value,self.server.application.token)

    def do_HEAD(self) -> None:
        if urlsplit(self.path).path in {'/api/video','/api/download'} and self.authorized():
            self.result(head=True);return
        self.headers_for(404,'text/plain',0)

    def do_GET(self) -> None:
        if not self.allowed_host():self.json(403,{'error':'僅允許本機存取。'});return
        path=urlsplit(self.path).path
        if path=='/':
            body=files('video_privacy').joinpath('assets/simple.html').read_text().replace('__LOCAL_TOKEN__',self.server.application.token).encode()
            self.headers_for(200,'text/html; charset=utf-8',len(body),
                             {'Set-Cookie':f'vpf_session={self.server.application.token}; HttpOnly; SameSite=Strict; Path=/'})
            self.wfile.write(body);return
        assets={'/app.js':('simple.js','text/javascript; charset=utf-8'),
                '/app.css':('simple.css','text/css; charset=utf-8')}
        assets['/simple.js']=assets['/app.js']
        assets['/simple.css']=assets['/app.css']
        if path in assets:
            name,mime=assets[path]
            body=files('video_privacy').joinpath('assets/'+name).read_bytes()
            self.headers_for(200,mime,len(body));self.wfile.write(body);return
        if not self.authorized():self.json(403,{'error':'請從本機處理頁面重新開啟。'});return
        if path=='/api/state':self.json(200,self.server.application.snapshot());return
        if path in {'/api/video','/api/download'}:self.result();return
        self.json(404,{'error':'找不到此頁面。'})

    def result(self,head: bool=False) -> None:
        application=self.server.application
        source=application.output
        if application.snapshot()['phase']!='done' or source is None or not source.is_file():
            self.json(404,{'error':'尚無已完成的影片。'});return
        size=source.stat().st_size
        start,end,status=0,size-1,200
        requested=self.headers.get('Range')
        if requested:
            try:
                if not requested.startswith('bytes=') or ',' in requested:raise ValueError
                left,right=requested[6:].split('-',1)
                if left:
                    start=int(left);end=min(int(right),size-1) if right else size-1
                else:
                    tail=int(right)
                    if tail<=0:raise ValueError
                    start=max(0,size-tail)
                if not 0<=start<=end<size:raise ValueError
                status=206
            except ValueError:
                self.headers_for(416,'video/mp4',0,{'Content-Range':f'bytes */{size}'});return
        headers={'Accept-Ranges':'bytes'}
        if status==206:headers['Content-Range']=f'bytes {start}-{end}/{size}'
        if urlsplit(self.path).path=='/api/download':headers['Content-Disposition']='attachment; filename="processed-video.mp4"'
        self.headers_for(status,'video/mp4',end-start+1,headers)
        if head:return
        try:
            with source.open('rb') as video:
                video.seek(start);remaining=end-start+1
                while remaining:
                    chunk=video.read(min(1024*1024,remaining))
                    if not chunk:break
                    self.wfile.write(chunk);remaining-=len(chunk)
        except (BrokenPipeError,ConnectionResetError):
            pass

    def do_POST(self) -> None:
        app=self.server.application
        token=self.headers.get('X-Local-Token','')
        origin=self.headers.get('Origin')
        expected={f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'}
        if not self.authorized() or not secrets.compare_digest(token,app.token) or (origin is not None and origin not in expected):
            self.close_connection=True
            self.json(403,{'error':'請從本機處理頁面操作。'});return
        if urlsplit(self.path).path!='/api/upload':
            self.close_connection=True;self.json(404,{'error':'找不到此操作。'});return
        try:length=int(self.headers.get('Content-Length','0'))
        except ValueError:length=0
        if not 0<length<=MAX_UPLOAD_BYTES:
            self.close_connection=True
            self.json(413,{'error':'請選擇不超過 2 GB 的影片。'});return
        extension=self.headers.get('X-Video-Extension','.mp4').lower()
        if extension not in {'.mp4','.mov','.m4v','.webm','.mkv','.avi'}:
            self.close_connection=True;self.json(400,{'error':'請選擇 MP4、MOV、WebM、MKV 或 AVI 影片。'});return
        try:source=app.reserve(extension)
        except RuntimeError:
            self.close_connection=True;self.json(409,{'error':'已有影片正在處理，請等待完成。'});return
        try:
            with source.open('xb') as output:
                remaining=length
                while remaining:
                    chunk=self.rfile.read(min(1024*1024,remaining))
                    if not chunk:raise ConnectionError
                    output.write(chunk);remaining-=len(chunk)
        except (ConnectionError,OSError):
            app.upload_failed(source)
            self.close_connection=True
            self.json(400,{'error':'影片未完整傳入，請重新選擇。'});return
        app.launch(source)
        self.json(202,{'accepted':True})


def serve(models: Path,checkpoint: Path | None=None,port: int=0,open_browser: bool=False) -> None:
    application=Application(models,checkpoint)
    try:
        server=LocalServer(('127.0.0.1',port),application)
    except OSError as exc:
        application.close()
        if port and exc.errno==errno.EADDRINUSE:
            url=f'http://127.0.0.1:{port}/'
            same_version=False
            try:
                with urlopen(url,timeout=2) as response:
                    same_version=(response.headers.get('X-Video-Privacy-Server')=='local-vpf'
                                  and response.headers.get('X-Video-Privacy-Version')==__version__)
            except Exception:
                pass
            if same_version:
                print(f'已開啟現有影片處理頁：{url}',flush=True)
                if open_browser:webbrowser.open(url)
                return
            raise ValueError(f'本機連接埠 {port} 已被占用；請關閉先前啟動的影片處理程式後再試。') from exc
        raise
    url=f'http://127.0.0.1:{server.server_port}/'
    print(f'影片隱私處理 {__version__}：{url}',flush=True)
    print('請保持此程式開啟；完成後在頁面下載影片。按 Ctrl+C 可關閉。',flush=True)
    if open_browser:webbrowser.open(url)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        server.server_close();application.close()
