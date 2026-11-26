from __future__ import annotations

import argparse
from importlib.resources import files
import json
import os
from pathlib import Path
import sys

from . import __version__


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(prog='vpf', description='本機影片文字遮罩；輸出後仍需人工檢查。')
    cli.add_argument('--version', action='version', version=f'vpf {__version__}')
    sub = cli.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare-models', help='下載官方權重與分詞器；只需準備一次')
    prepare.add_argument('--models', type=Path, default=Path('models'))
    prepare.add_argument('--checkpoint', type=Path, help='重用已下載的官方原生 checkpoint')
    inspect = sub.add_parser('inspect', help='檢查影片尺寸、時間與音軌數，不匯出畫面')
    inspect.add_argument('input', type=Path)
    editor = sub.add_parser('config-editor', help='匯出離線設定頁，可在瀏覽器圈選通知與表單區域')
    editor.add_argument('--output', type=Path, default=Path('config-editor.html'))
    demo = sub.add_parser('make-demo', help='建立虛構中英測試影片及對應設定')
    demo.add_argument('--output-dir', type=Path, default=Path('demo'))
    serve=sub.add_parser('serve',help='開啟簡易介面：放入影片、開始處理、下載結果')
    serve.add_argument('--models',type=Path,default=Path('models'))
    serve.add_argument('--checkpoint',type=Path)
    serve.add_argument('--port',type=int,default=0)
    serve.add_argument('--open-browser',action='store_true')
    redact = sub.add_parser('redact', help='逐畫面文字辨識與遮罩，輸出 MP4 與不含原文的報告')
    redact.add_argument('input', type=Path)
    redact.add_argument('--config', type=Path, help='進階手動設定；省略時自動辨識')
    redact.add_argument('--output', type=Path, required=True)
    redact.add_argument('--report', type=Path)
    redact.add_argument('--models', type=Path, default=Path('models'))
    redact.add_argument('--checkpoint', type=Path, help='重用已下載的官方原生 checkpoint')
    redact.add_argument('--audio', choices=['drop', 'keep'], default='drop',
                        help='drop=移除音軌；keep=保留所有原音軌，不遮蔽說話內容')
    redact.add_argument('--allow-no-notification-zone', action='store_true',
                        help='僅在此影片確定不需遮通知時使用')
    return cli


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == 'prepare-models':
            from .models import prepare
            payload = prepare(args.models.resolve(), args.checkpoint)
        elif args.command == 'inspect':
            from .video import inspect_video
            payload = inspect_video(args.input)
        elif args.command == 'config-editor':
            with args.output.open('xb') as target:
                target.write(files('video_privacy').joinpath('assets/config-editor.html').read_bytes())
            payload = {'editor':str(args.output), 'network':'none; local video stays in browser'}
        elif args.command == 'make-demo':
            from .demo import make_demo
            payload = make_demo(args.output_dir)
        elif args.command == 'serve':
            from .web import serve
            serve(args.models,args.checkpoint,args.port,args.open_browser)
            return 0
        elif args.config is None:
            if args.audio!='drop':
                raise ValueError('自動模式預設移除音軌；保留音軌請使用進階手動模式。')
            from .processing import run_automatic
            payload=run_automatic(args.input,args.output,args.models,args.checkpoint,report_path=args.report,
                                  progress=lambda count: print(f'已處理 {count} 個畫面',file=sys.stderr,flush=True) if count==1 or count%10==0 else None)
        else:
            from .config import load_config
            from .models import check_models, checkpoint_path
            config = load_config(args.config)
            if not any(r.kind == 'notification' for r in config.regions) and not args.allow_no_notification_zone:
                raise ValueError('尚未設定通知區域；請先圈選，或確認影片無通知後使用 --allow-no-notification-zone。')
            models = args.models.resolve()
            checkpoint = checkpoint_path(models, args.checkpoint)
            check_models(models, checkpoint)
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
            os.environ['TIKTOKEN_CACHE_DIR'] = str(models/'tokenizer')
            from .detection import Detector, PrivacyClassifier
            from .ocr import EasyOCRReader
            from .video import process_video
            classifier = PrivacyClassifier(str(checkpoint))
            report = args.report or args.output.with_suffix('.report.json')
            def progress(count: int) -> None:
                if count == 1 or count % 10 == 0:
                    print(f'已處理 {count} 個畫面', file=sys.stderr, flush=True)
            with EasyOCRReader(models, config) as ocr:
                payload = process_video(args.input, args.output, report, config,
                                        Detector(config, classifier), ocr, args.audio, progress)
        print(json.dumps(payload,ensure_ascii=False,indent=2))
        return 0
    except KeyboardInterrupt:
        print('已停止；未完成的結果不會交付。',file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, FileExistsError, FileNotFoundError) as exc:
        print(f'無法完成：{exc}',file=sys.stderr)
        return 2
    except Exception as exc:
        print(f'處理失敗（{type(exc).__name__}）；請檢查安裝與影片格式。未輸出原始文字。',file=sys.stderr)
        return 3


if __name__ == '__main__':
    raise SystemExit(main())
