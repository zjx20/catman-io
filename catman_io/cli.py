"""命令行：catman-io devices | setup | wake | wake-file"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from catman_io.config import Config


def _add_wake_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-c", "--config", type=Path, help="YAML 配置（见 config.example.yaml）")
    p.add_argument("-m", "--model", action="append", dest="models", help="唤醒词模型 ONNX，可多次指定")
    p.add_argument("-t", "--threshold", type=float, help="触发阈值（默认 0.5）")
    p.add_argument("--patience", type=int, help="连续多少帧高于阈值才触发（默认 1）")
    p.add_argument("--cooldown", type=float, help="触发后冷却秒数（默认 2）")
    p.add_argument("--vad", type=float, help="silero VAD 阈值，>0 启用（默认关闭）")


def _detector_from(args) -> tuple[Config, object]:
    from catman_io.wakeword import WakeWordDetector

    cfg = Config.load(args.config)
    w = cfg.wakeword
    models = args.models or w.models or None
    det = WakeWordDetector(
        models,
        threshold=args.threshold if args.threshold is not None else w.threshold,
        patience=args.patience if args.patience is not None else w.patience,
        cooldown=args.cooldown if args.cooldown is not None else w.cooldown,
        vad_threshold=args.vad if args.vad is not None else w.vad_threshold,
    )
    return cfg, det


def cmd_devices(args) -> int:
    from catman_io.audio.capture import list_devices

    print(list_devices())
    return 0


def cmd_setup(args) -> int:
    from catman_io.wakeword import bundled_models, ensure_base_models

    ensure_base_models()
    models = bundled_models()
    print(
        "base models ready;",
        f"{len(models)} bundled wake-word model(s):" if models else "no bundled wake-word model",
    )
    for m in models:
        print("  ", m)
    return 0


def cmd_wake(args) -> int:
    from catman_io.audio.capture import MicCapture

    cfg, det = _detector_from(args)
    a = cfg.audio
    device = args.device if args.device is not None else a.device
    channel = args.channel if args.channel is not None else a.channel
    print(f"listening for {', '.join(det.names)}  (threshold {det.threshold}, Ctrl-C to stop)")
    try:
        with MicCapture(device=device, channel=channel, sample_rate=a.sample_rate) as mic:
            for frame in mic.frames():
                fired = det.process(frame)
                score = max(det.last_scores.values()) if det.last_scores else 0.0
                bar = "#" * int(score * 30)
                sys.stdout.write(f"\r{score:5.2f} |{bar:<30}| dropped={mic.dropped}   ")
                for d in fired:
                    sys.stdout.write(f"\n[{time.strftime('%H:%M:%S')}] WAKE {d.model} score={d.score:.2f}\n")
                sys.stdout.flush()
    except KeyboardInterrupt:
        print()
    return 0


def cmd_wake_file(args) -> int:
    from catman_io.audio.frames import read_wav

    _, det = _detector_from(args)
    audio = read_wav(args.wav, channel=args.channel or 0)
    scores: list[float] = []
    dets = []
    from catman_io.audio.frames import iter_frames

    for frame in iter_frames(audio):
        dets += det.process(frame)
        scores.append(max(det.last_scores.values()))
    print(f"{args.wav}: {len(audio) / 16000:.2f}s, max score {max(scores):.3f}, {len(dets)} detection(s)")
    for d in dets:
        print(f"  t={d.stream_time:6.2f}s  {d.model}  score={d.score:.3f}")
    if args.scores:
        for i, s in enumerate(scores):
            print(f"  {i * 0.08:6.2f}s {s:.3f} {'#' * int(s * 40)}")
    return 0 if dets or not args.expect else 1


def cmd_webdemo(args) -> int:
    from catman_io.webdemo.server import run

    cfg = Config.load(args.config)
    w = cfg.wakeword
    run(
        host=args.host,
        port=args.port,
        model_paths=args.models or w.models or None,
        threshold=args.threshold if args.threshold is not None else w.threshold,
        patience=args.patience if args.patience is not None else w.patience,
        cooldown=args.cooldown if args.cooldown is not None else w.cooldown,
        vad_threshold=args.vad if args.vad is not None else w.vad_threshold,
        record_dir=args.record_dir,
        open_browser=args.open,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="catman-io", description="catman 语音输入输出软件栈")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("devices", help="列出音频设备").set_defaults(fn=cmd_devices)
    sub.add_parser("setup", help="下载 openWakeWord 基础模型并检查随包模型").set_defaults(fn=cmd_setup)

    p = sub.add_parser("wake", help="用麦克风实时检测唤醒词")
    _add_wake_args(p)
    p.add_argument("-d", "--device", help="输入设备编号或名字子串")
    p.add_argument("--channel", type=int, help="多声道设备取哪一路")
    p.set_defaults(fn=cmd_wake)

    p = sub.add_parser("wake-file", help="对一个 WAV 文件离线检测（测试录音）")
    _add_wake_args(p)
    p.add_argument("wav", type=Path)
    p.add_argument("--channel", type=int, help="多声道文件取哪一路")
    p.add_argument("--scores", action="store_true", help="逐帧打印分数")
    p.add_argument("--expect", action="store_true", help="没检测到时返回非零退出码（用于测试）")
    p.set_defaults(fn=cmd_wake_file)

    p = sub.add_parser("webdemo", help="网页版测试：浏览器麦克风实时看唤醒词命中，并可保存录音做训练样本")
    _add_wake_args(p)
    p.add_argument("--host", default="127.0.0.1", help="监听地址（浏览器只允许 localhost 或 https 用麦克风）")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--record-dir", type=Path, default=Path("data/recordings"), help="保存录音的目录")
    p.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    p.set_defaults(fn=cmd_webdemo)

    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.cmd == "wake" and getattr(args, "device", None) is not None and str(args.device).isdigit():
        args.device = int(args.device)
    return args.fn(args)
