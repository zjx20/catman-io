"""命令行入口：python -m training.wakeword <step> --config <yaml>"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

from .config import TrainingConfig

STEPS = ("synth", "resources", "features", "train", "evaluate", "install", "all")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m training.wakeword", description=__doc__)
    ap.add_argument("step", choices=STEPS)
    ap.add_argument("--config", required=True, help="训练配置 YAML")
    ap.add_argument("--overwrite", action="store_true", help="features：重新生成已存在的特征文件")
    ap.add_argument("--model", type=Path, help="evaluate/install：指定 ONNX 路径（默认用 export 目录里的）")
    ap.add_argument("--ncpu", type=int, default=0, help="features：特征计算线程数（默认 CPU 数的一半）")
    ap.add_argument(
        "--target", type=Path, default=Path("catman_io/wakeword/models"), help="install：模型安装目录"
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = TrainingConfig.load(args.config)
    cfg.work.mkdir(parents=True, exist_ok=True)

    steps = ["synth", "resources", "features", "train", "evaluate"] if args.step == "all" else [args.step]
    resources: dict[str, Path | None] = {"rir_dir": None, "validation": None, "precomputed": None}
    for step in steps:
        if step == "synth":
            from .tts import run_synthesis

            run_synthesis(cfg)
        elif step == "resources":
            from .resources import prepare_resources

            resources = prepare_resources(cfg)
        elif step in ("features", "train", "evaluate"):
            from .resources import prepare_resources

            resources = prepare_resources(cfg)  # 已下载的直接返回路径
            if step == "features":
                from .features import run_features

                run_features(cfg, resources["rir_dir"], ncpu=args.ncpu, overwrite=args.overwrite)
            elif step == "train":
                from .train import run_training

                run_training(cfg, resources["precomputed"], resources["validation"])
            else:
                from .evaluate import run_evaluation

                run_evaluation(cfg, args.model, resources["validation"], resources["rir_dir"])
        elif step == "install":
            src = args.model or cfg.export_dir / f"{cfg.model_name}.onnx"
            if not src.exists():
                print(f"model not found: {src}", file=sys.stderr)
                return 1
            args.target.mkdir(parents=True, exist_ok=True)
            for suffix in (".onnx", ".json"):
                p = src.with_suffix(suffix)
                if p.exists():
                    shutil.copy2(p, args.target / p.name)
                    print(f"installed {args.target / p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
