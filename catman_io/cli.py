"""命令行入口（子命令见 catman-io --help）。"""

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
    from catman_io.wakeword.detector import MODELS_DIR

    ensure_base_models()
    models = bundled_models()
    print(
        "base models ready;",
        f"{len(models)} bundled wake-word model(s):" if models else "no bundled wake-word model",
    )
    for m in models:
        print("  ", m)
    installed = MODELS_DIR / "INSTALLED"
    if installed.exists():
        print("   version:", installed.read_text(encoding="utf-8").strip())
    if not models:
        print("   fetch one with: python scripts/wakeword_model.py pull (see catman_io/wakeword/models/)")
    if args.list_asr:
        from catman_io.asr.models import ASR_MODELS

        for m in ASR_MODELS.values():
            print(f"  asr model {m.name:<22} {m.size_mb:>4} MB  {m.kind:<10} {m.note}")
    if args.asr is not None:
        from catman_io.asr.models import ensure_asr_model, resolve_model

        cfg = Config.load(args.config)
        model = resolve_model(cfg.asr.backend, args.asr or cfg.asr.model)
        path = ensure_asr_model(cfg.asr_model_dir, model)
        print(f"asr model {model.name} ready: {path}")
    return 0


def cmd_asr(args) -> int:
    """对 WAV 文件离线识别，顺带打印耗时与 RTF。"""
    from catman_io.asr import create_recognizer
    from catman_io.audio.frames import read_wav

    cfg = Config.load(args.config)
    if args.backend:
        cfg.asr.backend = args.backend
    if args.model:
        cfg.asr.model = args.model
    if args.threads:
        cfg.asr.threads = args.threads
    rec = create_recognizer(cfg)
    rc = 0
    for wav in args.wav:
        tr = rec.transcribe(read_wav(wav, channel=args.channel or 0))
        stats = f"{tr.duration:.2f}s audio, {tr.elapsed:.2f}s, RTF {tr.rtf:.2f}, lang {tr.language}"
        print(f"{wav}: {tr.text!r}  ({stats})")
        if not tr.text:
            rc = 1
    return rc


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


def cmd_listen(args) -> int:
    """唤醒 → 端点检测 → 存整句 WAV → 识别。在设备上验证前半段管线用。"""
    import wave

    from catman_io.audio.frames import FRAME_SECONDS
    from catman_io.dialog import Dialog, ReplyDone, StartTurn, State, Transcribe, TurnEnded
    from catman_io.vad import SileroVAD

    cfg, det = _detector_from(args)
    vad = SileroVAD(threads=cfg.vad.threads)
    dialog = Dialog(cfg.dialog, cfg.vad)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    asr = None
    if not args.no_asr and cfg.asr.backend != "none":
        try:
            from catman_io.asr import create_recognizer

            asr = create_recognizer(cfg)
        except Exception as e:  # noqa: BLE001
            print(f"asr disabled: {e}", file=sys.stderr)
    saved = 0
    frames_seen = 0
    mic = None
    wav_frames = None
    if args.wav:
        from catman_io.pipeline import WavFrames

        wav_frames = WavFrames(args.wav, channel=args.channel or 0)
        frames = iter(wav_frames)
        clock = lambda: frames_seen * FRAME_SECONDS  # noqa: E731
        max_frames = int(args.max_seconds / FRAME_SECONDS) if args.max_seconds else None
    else:
        from catman_io.audio.capture import MicCapture

        a = cfg.audio
        device = args.device if args.device is not None else a.device
        channel = args.channel if args.channel is not None else a.channel
        mic = MicCapture(device=device, channel=channel, sample_rate=a.sample_rate).start()
        frames = mic.frames()
        clock = time.monotonic
        max_frames = None
    print(f"listening for {', '.join(det.names)}; utterances go to {out_dir}  (Ctrl-C to stop)")
    try:
        for frame in frames:
            frames_seen += 1
            now = clock()
            dets = det.process(frame)
            prob = vad.process(frame)
            for cmd in dialog.on_frame(now, frame, dets, prob):
                if isinstance(cmd, StartTurn):
                    sys.stdout.write(f"\n[{time.strftime('%H:%M:%S')}] WAKE score={cmd.turn.wake_score}\n")
                elif isinstance(cmd, Transcribe):
                    turn = cmd.turn
                    assert turn.audio is not None
                    path = out_dir / f"{turn.id}.wav"
                    with wave.open(str(path), "wb") as w:
                        w.setnchannels(1)
                        w.setsampwidth(2)
                        w.setframerate(16000)
                        w.writeframes(turn.audio.tobytes())
                    saved += 1
                    dur = len(turn.audio) / 16000
                    sys.stdout.write(f"\n  utterance {dur:.2f}s -> {path}\n")
                    if asr is not None:
                        t0 = time.monotonic()
                        tr = asr.transcribe(turn.audio)
                        sys.stdout.write(f"  asr ({time.monotonic() - t0:.2f}s): {tr.text!r}\n")
                    for c2 in dialog.on_event(now, ReplyDone(turn.id)):
                        assert isinstance(c2, TurnEnded)
                elif isinstance(cmd, TurnEnded):
                    sys.stdout.write(f"\n  turn ended: {cmd.status}\n")
            score = max(det.last_scores.values()) if det.last_scores else 0.0
            state = dialog.state.value
            sys.stdout.write(f"\r{state:<9} wake={score:4.2f} vad={prob:4.2f} |{'#' * int(prob * 20):<20}| ")
            sys.stdout.flush()
            if max_frames is not None and frames_seen >= max_frames:
                break
            if wav_frames is not None and wav_frames.exhausted and dialog.state == State.IDLE:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if mic is not None:
            mic.stop()
    print(f"\n{saved} utterance(s) saved")
    return 0


def cmd_run(args) -> int:
    """整条管线：唤醒 → 聆听 → 识别 → 应答 → 播报。"""
    from catman_io.pipeline import build_pipeline

    cfg = Config.load(args.config)
    if args.threshold is not None:
        cfg.wakeword.threshold = args.threshold
    if args.device is not None:
        cfg.audio.device = args.device
    if args.channel is not None:
        cfg.audio.channel = args.channel
    if args.no_tts:
        cfg.tts.backend = "none"
    if args.no_asr:
        cfg.asr.backend = "none"
    quiet = args.wav is not None and not args.status

    def status(line: str) -> None:
        if not quiet:
            sys.stdout.write(f"\r{line[:100]:<100}")
            sys.stdout.flush()

    def on_turn(turn, st: str) -> None:
        text = turn.info.get("asr_text", "")
        reply = turn.info.get("reply_text", "")
        sys.stdout.write(f"\n[{time.strftime('%H:%M:%S')}] {st:<9} 「{text}」 → 「{reply}」\n")

    pipe = build_pipeline(
        cfg,
        input_wav=args.wav,
        output_wav=args.out,
        realtime=args.realtime,
        max_seconds=args.max_seconds,
        on_status=status,
        on_turn=on_turn,
        model_paths=args.models,
        api=not args.no_api and args.wav is None,
    )
    if pipe.api is not None:
        print(f"api: http://{cfg.api.host}:{cfg.api.port}/api/health  (token: {cfg.api_token_path})")
    if args.wav is None:
        print("running; say the wake word (Ctrl-C to stop)")
    try:
        pipe.run()
    except KeyboardInterrupt:
        pipe.stop()
    print(f"\n{pipe.turns_done} turn(s)")
    return 0


def cmd_intent(args) -> int:
    """意图规则：parse 一句话 / test 跑回归用例 / lint 自检 / list 列意图。"""
    import json

    from catman_io.intent import load_rules
    from catman_io.intent.normalize import normalize

    cfg = Config.load(args.config)
    rs = load_rules(cfg)
    if args.sub == "parse":
        text = " ".join(args.text)
        print(f"normalized: {normalize(text)}")
        hits = rs.match_all(text)
        if not hits:
            print("rule: (no match)")
        for i, h in enumerate(hits):
            mark = "rule:" if i == 0 else "     "
            print(f"{mark} {h.name} {json.dumps(h.slots, ensure_ascii=False)}  [{h.rule_id}] {h.raw!r}")
        if args.llm:
            from catman_io.intent.router import build_router

            router = build_router(cfg, rs)
            res = router.route(text, use_rules=False)
            print(f"llm: {res.intent.to_dict() if res.intent else None}  ({res.elapsed_llm:.2f}s)")
        return 0
    if args.sub == "lint":
        errors, warnings = rs.lint()
        for w in warnings:
            print(f"warning: {w}")
        for e in errors:
            print(f"error: {e}")
        print(
            f"{len(rs.intents)} intents, {len(rs.rules)} patterns, "
            f"{len(errors)} error(s), {len(warnings)} warning(s)"
        )
        return 1 if errors else 0
    if args.sub == "test":
        from catman_io.intent.cases import evaluate, read_cases

        errors, _ = rs.lint()
        for e in errors:
            print(f"lint error: {e}")
        cases = read_cases(args.cases or cfg.cases_path)
        report = evaluate(rs, cases)
        print(report.format())
        return 0 if report.ok and not errors else 1
    if args.sub == "list":
        for spec in rs.intents.values():
            slots = " ".join(f"{{{k}:{v}}}" for k, v in spec.slots.items())
            print(f"{spec.name:<16} {slots:<24} {spec.description}  ({len(spec.patterns)} patterns)")
        return 0
    return 2


def _parse_since(text: str | None) -> float | None:
    """1d / 12h / 30m 或 YYYY-MM-DD → 时间戳。"""
    import datetime as dt
    import re

    if not text:
        return None
    m = re.fullmatch(r"(\d+)([dhm])", text)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        seconds = n * {"d": 86400, "h": 3600, "m": 60}[unit]
        return time.time() - seconds
    return dt.datetime.strptime(text, "%Y-%m-%d").timestamp()


def cmd_journal(args) -> int:
    """回合日志：list 列最近回合 / show 看一条 / review 出 bad case 复盘（Markdown）。"""
    import datetime as dt
    import json

    from catman_io.journal import BAD_FLAGS, Journal
    from catman_io.journal.review import build_review, render_markdown, write_markdown

    cfg = Config.load(args.config)
    journal = Journal(cfg.journal_dir, save_audio=False, keep_days=0)
    if args.sub == "list":
        flags = {args.flag} if args.flag else (BAD_FLAGS if args.bad else None)
        recs = journal.records(since=_parse_since(args.since), flags=flags, limit=args.limit)
        for r in recs:
            t = dt.datetime.fromtimestamp(r.get("at", 0)).strftime("%m-%d %H:%M:%S")
            intent = f"{r.get('intent')}({r.get('tier')})" if r.get("intent") else "-"
            lat_ms = r.get("lat_first_audio_ms")
            lat = f"{lat_ms / 1000:.1f}s" if lat_ms is not None else "   -"
            fl = ",".join(r.get("flags") or [])
            reply = (r.get("reply_text") or "")[:40]
            text = r.get("text") or ""
            head = f"{t} {r['turn_id']} {r.get('status', ''):<9} {lat:>5}"
            print(f"{head} 「{text}」 → {intent} 「{reply}」 {fl}")
        print(f"{len(recs)} turn(s)")
        return 0
    if args.sub == "show":
        r = journal.get(args.turn_id)
        if r is None:
            print("not found", file=sys.stderr)
            return 1
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return 0
    if args.sub == "review":
        from catman_io.intent import load_rules

        intents = {name: spec.description for name, spec in load_rules(cfg).intents.items()}
        since = _parse_since(args.since)
        review = build_review(journal, since=since, include_acked=args.all, intents=intents)
        if args.output:
            print(f"wrote {write_markdown(review, args.output)} ({review['count']} turn(s))")
        else:
            print(render_markdown(review))
        return 0
    return 2


def cmd_flywheel(args) -> int:
    """飞轮：export 导出复盘 / nudge 叫 catman 来处理 / ack 记下看到哪里。"""
    from catman_io.journal import Journal
    from catman_io.journal.review import build_review, write_ack, write_markdown

    cfg = Config.load(args.config)
    journal = Journal(cfg.journal_dir, save_audio=False, keep_days=0)
    if args.sub == "export":
        from catman_io.intent import load_rules

        intents = {name: spec.description for name, spec in load_rules(cfg).intents.items()}
        since = _parse_since(args.since)
        review = build_review(journal, since=since, include_acked=args.all, intents=intents)
        out = args.output or cfg.journal_dir / f"review-{time.strftime('%Y%m%d')}.md"
        print(f"wrote {write_markdown(review, out)} ({review['count']} turn(s), until={review['until']})")
        return 0
    if args.sub == "ack":
        write_ack(journal, args.until)
        print(f"acked until {args.until}")
        return 0
    if args.sub == "nudge":
        from catman_io.brain.catman import build_catman

        catman = build_catman(cfg)
        if catman is None:
            print("brain.catman.base_url is not configured", file=sys.stderr)
            return 1
        message = args.message or (
            "請按 catman-io 技能處理最新嘅語音意圖 bad case"
            "（拉複盤、改 site.yaml、跑回歸、寫回、ack），完成後簡短匯報。"
        )
        catman.post(message)
        print("nudged catman")
        return 0
    return 2


def cmd_say(args) -> int:
    """合成一段粤语并播放（或写 WAV），顺带看首句延迟。"""
    import numpy as np

    from catman_io.tts import clean_for_speech, create_synthesizer, split_sentences

    cfg = Config.load(args.config)
    if args.voice:
        cfg.tts.voice = args.voice
    if args.rate:
        cfg.tts.rate = args.rate
    if args.pitch:
        cfg.tts.pitch = args.pitch
    tts = create_synthesizer(cfg)
    if tts is None:
        print("tts.backend is 'none'", file=sys.stderr)
        return 1
    text = clean_for_speech(" ".join(args.text))
    sentences = split_sentences(text)
    if not sentences:
        print("nothing to say", file=sys.stderr)
        return 1
    speaker = None
    if args.output is None:
        from catman_io.tts.speaker import SoundDeviceOutput, Speaker

        output = SoundDeviceOutput(
            cfg.audio.output_device, blocksize=cfg.speaker.blocksize, latency=cfg.speaker.latency
        )
        speaker = Speaker(output, volume=cfg.speaker.volume, blocksize=cfg.speaker.blocksize)
        speaker.set_gen(1)
    pieces = []
    t_start = time.perf_counter()
    for s in sentences:
        pcm = tts.synthesize(s)
        stamp = f"+{time.perf_counter() - t_start:5.2f}s"
        cached = " (cached)" if getattr(tts, "last_fetch_seconds", 1.0) == 0.0 else ""
        print(f"{stamp} {len(pcm) / 16000:5.2f}s audio{cached}: {s}")
        if speaker is not None:
            speaker.say(pcm, gen=1)
        else:
            pieces.append(pcm)
    if speaker is not None:
        speaker.wait()
        speaker.close()
    else:
        import wave

        with wave.open(str(args.output), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(np.concatenate(pieces).tobytes())
        print(f"wrote {args.output}")
    return 0


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
    cfg_parent = argparse.ArgumentParser(add_help=False)
    cfg_parent.add_argument("-c", "--config", type=Path, help="YAML 配置")
    ap = argparse.ArgumentParser(prog="catman-io", description="catman 语音输入输出软件栈")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("devices", help="列出音频设备").set_defaults(fn=cmd_devices)
    p = sub.add_parser("setup", help="下载 openWakeWord 基础模型（--asr 再下载语音识别模型）")
    p.add_argument("-c", "--config", type=Path, help="YAML 配置（决定 ASR 后端与模型目录）")
    p.add_argument("--asr", nargs="?", const="", metavar="NAME", help="下载 ASR 模型；不给名字用配置里的默认")
    p.add_argument("--list-asr", action="store_true", help="列出可选的 ASR 模型")
    p.set_defaults(fn=cmd_setup)

    p = sub.add_parser("asr", help="对 WAV 文件离线做粤语识别（测试识别器）")
    p.add_argument("wav", type=Path, nargs="+")
    p.add_argument("-c", "--config", type=Path, help="YAML 配置")
    p.add_argument("--backend", choices=["sensevoice", "wenet_yue"], help="覆盖配置里的 asr.backend")
    p.add_argument("--model", help="覆盖配置里的 asr.model")
    p.add_argument("--threads", type=int, help="识别线程数")
    p.add_argument("--channel", type=int, help="多声道文件取哪一路")
    p.set_defaults(fn=cmd_asr)

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

    p = sub.add_parser("listen", help="唤醒后听一句话：端点检测切句、存成 WAV 并识别（验证前半段管线）")
    _add_wake_args(p)
    p.add_argument("-d", "--device", help="输入设备编号或名字子串")
    p.add_argument("--channel", type=int, help="多声道设备取哪一路")
    p.add_argument("--wav", type=Path, help="用 WAV 文件代替麦克风（离线测试）")
    p.add_argument("--out-dir", type=Path, default=Path("data/utterances"), help="整句录音保存目录")
    p.add_argument("--no-asr", action="store_true", help="只切句不识别")
    p.add_argument("--max-seconds", type=float, default=None, help="--wav 时最多跑多少秒")
    p.set_defaults(fn=cmd_listen)

    p = sub.add_parser("run", help="跑整条管线：唤醒 → 聆听 → 识别 → 应答 → 播报")
    _add_wake_args(p)
    p.add_argument("-d", "--device", help="输入设备编号或名字子串")
    p.add_argument("--channel", type=int, help="多声道设备取哪一路")
    p.add_argument("--wav", type=Path, help="用 WAV 文件代替麦克风（离线测试）")
    p.add_argument("--out", type=Path, help="把播出的声音写到这个 WAV 而不是扬声器")
    p.add_argument("--realtime", action="store_true", help="--wav 时按真实时间节奏喂")
    p.add_argument("--max-seconds", type=float, help="最多跑多少秒（按音频时钟）")
    p.add_argument("--status", action="store_true", help="--wav 时也打印状态行")
    p.add_argument("--no-tts", action="store_true", help="不合成不出声，只打日志")
    p.add_argument("--no-asr", action="store_true", help="不识别（只验证唤醒与切句）")
    p.add_argument("--no-api", action="store_true", help="不开本机 HTTP API")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("intent", help="意图规则：parse 一句话 / test 回归用例 / lint 自检 / list")
    ip = p.add_subparsers(dest="sub", required=True)
    q = ip.add_parser(parents=[cfg_parent], name="parse", help="看一句话命中哪条规则")
    q.add_argument("text", nargs="+")
    q.add_argument("--llm", action="store_true", help="也问一次 LLM（需要配置 intent.llm）")
    ip.add_parser(parents=[cfg_parent], name="lint", help="检查规则文件：例句必须命中、模式合法")
    q = ip.add_parser(parents=[cfg_parent], name="test", help="用回归用例跑规则")
    q.add_argument("--cases", type=Path, help="用例文件（默认配置里的）")
    ip.add_parser(parents=[cfg_parent], name="list", help="列出意图")
    p.set_defaults(fn=cmd_intent)

    p = sub.add_parser("journal", help="回合日志：list / show / review（bad case 复盘）")
    jp = p.add_subparsers(dest="sub", required=True)
    q = jp.add_parser(parents=[cfg_parent], name="list", help="列最近的回合")
    q.add_argument("--bad", action="store_true", help="只看有 bad case 标记的")
    q.add_argument("--flag", help="只看带这个标记的")
    q.add_argument("--since", help="1d / 12h / 30m 或 YYYY-MM-DD")
    q.add_argument("-n", "--limit", type=int, default=50)
    q = jp.add_parser(parents=[cfg_parent], name="show", help="看一条回合的完整记录")
    q.add_argument("turn_id")
    q = jp.add_parser(parents=[cfg_parent], name="review", help="输出 bad case 复盘（Markdown）")
    q.add_argument("--since", help="1d / 12h 或 YYYY-MM-DD")
    q.add_argument("--all", action="store_true", help="包括已经 ack 过的")
    q.add_argument("-o", "--output", type=Path, help="写到文件")
    p.set_defaults(fn=cmd_journal)

    p = sub.add_parser("flywheel", help="意图飞轮：export 复盘 / nudge 叫 catman / ack")
    fp = p.add_subparsers(dest="sub", required=True)
    q = fp.add_parser(parents=[cfg_parent], name="export", help="把 bad case 复盘写成 Markdown")
    q.add_argument("--since", help="1d / 12h 或 YYYY-MM-DD")
    q.add_argument("--all", action="store_true", help="包括已经 ack 过的")
    q.add_argument("-o", "--output", type=Path)
    q = fp.add_parser(parents=[cfg_parent], name="ack", help="记下复盘看到哪里（时间戳）")
    q.add_argument("until", type=float)
    q = fp.add_parser(parents=[cfg_parent], name="nudge", help="往 catman 发一句话，叫它按技能处理 bad case")
    q.add_argument("-m", "--message")
    p.set_defaults(fn=cmd_flywheel)

    p = sub.add_parser("say", help="合成一段粤语并播放（-o 写成 WAV），测试 TTS 与扬声器")
    p.add_argument("text", nargs="+")
    p.add_argument("-c", "--config", type=Path, help="YAML 配置")
    p.add_argument("-o", "--output", type=Path, help="不播放，写到这个 WAV")
    p.add_argument("--voice", help="覆盖 tts.voice，例如 zh-HK-WanLungNeural")
    p.add_argument("--rate", help="语速，例如 +10%%")
    p.add_argument("--pitch", help="音高，例如 -20Hz")
    p.set_defaults(fn=cmd_say)

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
    device = getattr(args, "device", None)
    if args.cmd in ("wake", "listen", "run") and device is not None and str(device).isdigit():
        args.device = int(args.device)
    return args.fn(args)
