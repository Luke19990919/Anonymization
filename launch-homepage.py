"""Start a detached local service and open its homepage after readiness."""
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import webbrowser
from video_privacy import __version__

URL = 'http://127.0.0.1:8765/'


def probe() -> bool:
    try:
        with urllib.request.urlopen(URL, timeout=1) as response:
            if response.headers.get('X-Video-Privacy-Server') != 'local-vpf':
                raise RuntimeError('連接埠 8765 被其他程式占用。')
            if response.headers.get('X-Video-Privacy-Version') != __version__:
                raise RuntimeError('舊版服務仍在執行，請先關閉舊版服務或重新開機。')
            return True
    except urllib.error.URLError:
        return False


def main() -> None:
    root = Path(__file__).resolve().parent
    if probe():
        webbrowser.open(URL)
        print('已開啟現有首頁。')
        return
    runtime = root / '.runtime'
    runtime.mkdir(exist_ok=True, mode=0o700)
    log_path = runtime / 'startup.log'
    with log_path.open('wb') as log:
        child = subprocess.Popen(
            [sys.executable, '-m', 'video_privacy.cli', 'serve',
             '--models', str(root / 'models'), '--port', '8765'],
            cwd=root, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
    for _ in range(100):
        if probe():
            webbrowser.open(URL)
            print('首頁已開啟；可以關閉此視窗。重開機後請再次雙擊啟動檔。')
            return
        if child.poll() is not None:
            raise RuntimeError(f'服務啟動失敗，請查看 {log_path}')
        time.sleep(0.1)
    child.terminate()
    raise RuntimeError(f'服務啟動逾時，請查看 {log_path}')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'啟動失敗：{exc}', file=sys.stderr)
        sys.exit(1)
