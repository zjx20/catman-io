#!/usr/bin/env python3
"""唤醒词模型的版本管理：代码 + 模型 + 训练数据放在独立的 ``models/wakeword/<版本>`` 分支里。

一个版本分支 = 训练它的那个代码提交 + 一个提交，后者装进：

- ``catman_io/wakeword/models/<模型名>.onnx`` / ``.json``（模型与元数据：训练数据量、配置、评估）
- ``training/wakeword/work/<名字>/``（合成片段、manifest、export 里的导出产物；
  ``resources/`` 这种可重复下载的、``features/`` 这种可重算的不进来）
- 配置里引用的真人录音 / 环境噪声目录（``data.extra_*_dirs``、``augment.background_dirs`` 等，
  录了就没法再下载，必须跟着版本走；目录要在仓库里）
- 训练用的配置 YAML 与根目录的 ``MODEL.md``（版本说明，自动生成）

主分支（代码）不带模型二进制，只有 ``catman_io/wakeword/models/VERSION`` 指向推荐版本。

    python scripts/wakeword_model.py list                 # 远端 / 本地有哪些版本
    python scripts/wakeword_model.py pull [v2]            # 只取模型文件到 catman_io/wakeword/models/
                                                          # （不给版本就取 VERSION 文件里写的）
    python scripts/wakeword_model.py show v2              # 打印某个版本的 MODEL.md
    python scripts/wakeword_model.py publish v3 --notes "改了什么、真机表现" --push
                                                          # 把当前代码 + 训练产物发布成新版本分支

想复现或在同样数据上重训：``git checkout models/wakeword/v2``，然后直接跑
``python -m training.wakeword features / train``。
``publish`` 用 git 底层命令直接造提交，不碰工作区、不切分支；``pull`` 尽量用部分克隆（只拉模型那两个 blob）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BRANCH_PREFIX = "models/wakeword/"
REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path("catman_io/wakeword/models")  # 相对仓库根目录
DEFAULT_CONFIG = Path("training/wakeword/configs/siu_maau_jan.yaml")
NOTES_FILE = "MODEL.md"
VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
MODEL_SUFFIXES = (".onnx", ".json")
SKIP_WORK_DIRS = {"resources", "features"}  # 可重复下载 / 可重算


class Git:
    def __init__(self, repo: Path, env: dict[str, str] | None = None):
        self.repo = repo
        self.env = {**os.environ, **(env or {})}

    def __call__(self, *args: str, check: bool = True, input: str | None = None) -> str:
        r = subprocess.run(
            ["git", *args],
            cwd=self.repo,
            env=self.env,
            capture_output=True,
            text=True,
            input=input,
            check=False,
        )
        if check and r.returncode != 0:
            raise SystemExit(f"git {' '.join(args)} failed:\n{r.stderr.strip()}")
        return r.stdout

    def bytes(self, *args: str) -> bytes:
        r = subprocess.run(["git", *args], cwd=self.repo, env=self.env, capture_output=True, check=True)
        return r.stdout

    def ok(self, *args: str) -> bool:
        r = subprocess.run(["git", *args], cwd=self.repo, env=self.env, capture_output=True, check=False)
        return r.returncode == 0


def branch_name(version: str) -> str:
    if not VERSION_RE.match(version):
        raise SystemExit(f"bad version {version!r}: use letters / digits / . _ - (e.g. v2, v2.1)")
    return BRANCH_PREFIX + version


def remote_versions(git: Git, remote: str) -> dict[str, str]:
    out = git("ls-remote", "--heads", remote, f"refs/heads/{BRANCH_PREFIX}*", check=False)
    prefix = f"refs/heads/{BRANCH_PREFIX}"
    versions = {}
    for line in out.splitlines():
        sha, _, ref = line.partition("\t")
        if ref.startswith(prefix):
            versions[ref[len(prefix) :]] = sha
    return versions


def local_versions(git: Git) -> dict[str, str]:
    out = git(
        "for-each-ref",
        "--format=%(objectname) %(refname)",
        f"refs/heads/{BRANCH_PREFIX}",
        f"refs/remotes/*/{BRANCH_PREFIX}",
    )
    versions: dict[str, str] = {}
    for line in out.splitlines():
        sha, _, ref = line.partition(" ")
        versions.setdefault(ref.rsplit("/", 1)[-1], sha)
    return versions


def resolve_ref(git: Git, version: str, remote: str) -> str:
    """从远端取这个版本的分支（优先部分克隆，只拉提交和树）；取不到就退回本地已有的引用。"""
    branch = branch_name(version)
    tracking = f"refs/remotes/{remote}/{branch}"
    refspec = f"+refs/heads/{branch}:{tracking}"
    if git.ok("fetch", "--no-tags", "--filter=blob:none", remote, refspec) or git.ok(
        "fetch", "--no-tags", remote, refspec
    ):
        return tracking
    for ref in (tracking, f"refs/heads/{branch}"):
        if git.ok("rev-parse", "--verify", "--quiet", ref):
            print(f"warning: could not fetch {branch} from {remote}; using local {ref}", file=sys.stderr)
            return ref
    raise SystemExit(f"model version {version!r} not found: no branch {branch} on {remote} or locally")


def default_version(repo: Path) -> str:
    p = repo / MODELS_DIR / "VERSION"
    if not p.exists():
        raise SystemExit(f"no version given and {p} does not exist")
    return p.read_text(encoding="utf-8").strip()


def _version_key(v: str):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", v)]


# ---------------------------------------------------------------- 摘要 / 说明
def _get(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def summary_lines(meta: dict) -> list[str]:
    """从模型的 .json 元数据里挑出最值得看的几行。字段缺失时跳过（老版本的元数据没有快语速测试）。"""
    lines = []
    cfg = meta.get("config", {})
    data = meta.get("data", {})
    if meta.get("trained_at"):
        lines.append(f"训练时间：{meta['trained_at']}，{meta.get('training_seconds', '?')} s")
    if data:
        real = data.get("real_recordings") or {}
        real_note = (
            f"其中真人录音 {real.get('positive_train')} / {real.get('negative_train')}，"
            if real.get("positive_train") or real.get("negative_train")
            else ""
        )
        lines.append(
            f"训练数据：正样本 {data.get('positive_train')} / 负样本 {data.get('negative_train')}（增强后，"
            f"{real_note}通用负样本 {data.get('precomputed_negatives')} 段），"
            f"误唤醒验证集 {data.get('fp_validation_hours')} h"
        )
    tts, aug, train = cfg.get("tts", {}), cfg.get("augment", {}), cfg.get("train", {})
    if tts:
        lines.append(
            f"合成语速：正样本 {' '.join(tts.get('rates', []))}；"
            f"负样本 {' '.join(tts.get('negative_rates', []))}；音色 {len(tts.get('voices', []))} 个"
        )
    if aug:
        speed = f"重采样变速 p={aug.get('p_speed')} {aug.get('speed_factors')}"
        tempo = f"WSOLA 变速 p={aug.get('p_tempo')} {aug.get('tempo_range')}" if "p_tempo" in aug else ""
        lines.append("后处理变速：" + "；".join(x for x in (speed, tempo) if x))
    if train:
        lines.append(
            f"分类器：{train.get('layer_size')} 宽 × {train.get('n_blocks')} 块，"
            f"dropout {train.get('dropout')}，{train.get('steps')} 步，"
            f"负样本权重上限 {train.get('max_negative_weight')}，"
            f"误唤醒目标 {train.get('target_fp_per_hour')}/h"
        )
    ev = meta.get("evaluation", {})
    if ev:
        parts = []
        for name, label in (
            ("clean", "干净"),
            ("augmented", "增强"),
            ("fast_x1.5", "1.5 倍速"),
            ("fast_x2.0", "2 倍速"),
        ):
            r = _get(ev, name, "recall@0.5")
            if r is not None:
                parts.append(f"{label} {r:.1%}")
        if parts:
            lines.append("召回@0.5：" + "，".join(parts))
        kinds = _get(ev, "clean", "false_accept_by_kind@0.5", default={})
        if kinds:
            lines.append("误接受@0.5：" + "，".join(f"{k} {v:.1%}" for k, v in kinds.items()))
        real = ev.get("real", {})
        if real.get("n_positive"):
            by_voice = "，".join(f"{k} {v:.1%}" for k, v in real.get("recall_by_voice@0.5", {}).items())
            lines.append(
                f"真人录音召回@0.5：{real['recall@0.5']:.1%}（{real['n_positive']} 条"
                + (f"；{by_voice}" if by_voice else "")
                + "）"
            )
        fv = ev.get("fp_validation", {})
        if fv:
            lines.append(
                f"通用音频误唤醒：{fv.get('activations_per_hour@0.5')}/h @0.5，"
                f"{fv.get('activations_per_hour@0.9')}/h @0.9（{fv.get('hours')} h）"
            )
    return lines


def render_notes(version: str, meta: dict, notes: str, code: dict, files: list[str]) -> str:
    phrase = meta.get("wake_phrase", "?")
    dirty = "，工作区当时有未提交改动" if code["dirty"] else ""
    lines = [
        f"# 唤醒词模型 {version}（{phrase}）",
        "",
        f"模型名 `{meta.get('model_name', '?')}`（openWakeWord 分类器）。这条分支 = 训练它的代码提交"
        f" {code['commit']} + 本提交（模型、训练数据、配置），由 `scripts/wakeword_model.py publish` 生成。",
        "",
        f"- 发布时间：{code['published_at']}",
        f"- 训练代码：commit {code['commit']}（分支 {code['branch']}{dirty}）",
    ]
    lines += [f"- {x}" for x in summary_lines(meta)]
    lines += ["", "## 说明", "", notes.strip() or "（无）", "", "## 本提交带的文件", ""]
    lines += [f"- `{f}`" for f in files]
    lines += [
        "",
        "## 使用",
        "",
        "```bash",
        f"python scripts/wakeword_model.py pull {version}     # 只取模型文件到 catman_io/wakeword/models/",
        f"git checkout {BRANCH_PREFIX}{version}                # 要复现 / 重训：代码、配置、合成片段都在",
        "python -m training.wakeword resources --config training/wakeword/configs/siu_maau_jan.yaml",
        "```",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- 子命令
def cmd_list(args) -> int:
    git = Git(args.repo)
    remote = remote_versions(git, args.remote)
    local = local_versions(git)
    versions = sorted(set(remote) | set(local), key=_version_key)
    if not versions:
        print(f"no model versions found (branches {BRANCH_PREFIX}* on {args.remote} or locally)")
        return 0
    try:
        current = default_version(args.repo)
    except SystemExit:
        current = ""
    for v in versions:
        where = " ".join(w for w, present in ((args.remote, v in remote), ("local", v in local)) if present)
        mark = "  <- VERSION" if v == current else ""
        print(f"{v:<12}{(remote.get(v) or local.get(v))[:12]}  {where}{mark}")
    return 0


def cmd_show(args) -> int:
    git = Git(args.repo)
    ref = resolve_ref(git, args.version, args.remote)
    print(git("cat-file", "blob", f"{ref}:{NOTES_FILE}"))
    return 0


def cmd_pull(args) -> int:
    git = Git(args.repo)
    version = args.version or default_version(args.repo)
    ref = resolve_ref(git, version, args.remote)
    sha = git("rev-parse", ref).strip()
    dest = args.dest if args.dest.is_absolute() else args.repo / args.dest
    dest.mkdir(parents=True, exist_ok=True)
    listing = git("ls-tree", "--name-only", f"{ref}:{MODELS_DIR.as_posix()}", check=False)
    files = [f for f in listing.splitlines() if f.endswith(MODEL_SUFFIXES)]
    if not files:
        raise SystemExit(f"branch {branch_name(version)} has no model files under {MODELS_DIR}")
    for stale in dest.glob("*"):
        if stale.suffix in MODEL_SUFFIXES and stale.name not in files:
            stale.unlink()  # 换版本时清掉旧模型，别让两个模型同时被加载
    for name in files:
        (dest / name).write_bytes(git.bytes("cat-file", "blob", f"{ref}:{MODELS_DIR.as_posix()}/{name}"))
    (dest / "INSTALLED").write_text(f"{version} {sha}\n", encoding="utf-8")
    print(f"installed wake-word model {version} ({sha[:12]}) -> {dest}: {', '.join(files)}")
    for name in files:
        if name.endswith(".json"):
            for line in summary_lines(json.loads((dest / name).read_text(encoding="utf-8"))):
                print("  " + line)
    return 0


# 配置里这些键指向的目录是录来的数据（真人录音、房间噪声、自己测的冲激响应），跟模型一起进版本分支
DATA_DIR_KEYS = (
    ("data", "extra_positive_dirs"),
    ("data", "extra_negative_dirs"),
    ("data", "extra_positive_val_dirs"),
    ("data", "extra_negative_val_dirs"),
    ("augment", "background_dirs"),
    ("augment", "rir_dirs"),
)


def _config_paths(config: Path) -> tuple[Path, list[Path]]:
    """训练配置里的 workdir 和它引用的数据目录（都是相对仓库根目录的相对路径，或绝对路径）。"""
    import yaml

    raw = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    workdir = Path(raw.get("workdir") or "training/wakeword/work/siu_maau_jan")
    dirs: list[Path] = []
    for section, key in DATA_DIR_KEYS:
        for d in (raw.get(section) or {}).get(key) or []:
            if Path(d) not in dirs:
                dirs.append(Path(d))
    return workdir, dirs


def _data_dir_files(repo: Path, dirs: list[Path]) -> dict[str, Path]:
    """{分支里的路径: 本地文件}。目录不在仓库里就带不走，报错让人先挪进来。"""
    files: dict[str, Path] = {}
    for d in dirs:
        local = d if d.is_absolute() else repo / d
        if not local.is_dir():
            raise SystemExit(f"data directory in config does not exist: {d}")
        try:
            rel = local.resolve().relative_to(repo.resolve())
        except ValueError:
            raise SystemExit(f"data directory {d} is outside the repository; move it inside first") from None
        for p in sorted(local.rglob("*")):
            if p.is_file():
                files[(rel / p.relative_to(local)).as_posix()] = p
    return files


def cmd_publish(args) -> int:
    git = Git(args.repo)
    branch = branch_name(args.version)
    config = args.config if args.config.is_absolute() else args.repo / args.config
    if not config.exists():
        raise SystemExit(f"config not found: {config}")
    workdir, data_dirs = _config_paths(config)
    workdir_abs = workdir if workdir.is_absolute() else args.repo / workdir
    export = args.src if args.src else workdir_abs / "export"
    export = export if export.is_absolute() else args.repo / export
    onnx = sorted(export.glob("*.onnx"))
    if len(onnx) != 1:
        raise SystemExit(f"expected exactly one .onnx in {export}, found {len(onnx)}")
    meta_path = onnx[0].with_suffix(".json")
    if not meta_path.exists():
        raise SystemExit(f"missing {meta_path} (train + evaluate write it next to the .onnx)")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if "evaluation" not in meta:
        print(f"warning: {meta_path.name} has no evaluation section; run `evaluate` first", file=sys.stderr)

    exists_local = git.ok("rev-parse", "--verify", "--quiet", f"refs/heads/{branch}")
    exists_remote = args.version in remote_versions(git, args.remote)
    if (exists_local or exists_remote) and not args.force:
        raise SystemExit(
            f"{branch} already exists (local={exists_local}, {args.remote}={exists_remote}); "
            "pick a new version or pass --force"
        )

    base = git("rev-parse", "--verify", args.base).strip()
    notes = Path(args.notes_file).read_text(encoding="utf-8") if args.notes_file else args.notes
    names = git("branch", "--points-at", base, "--format=%(refname:short)").split()
    code = {
        "commit": base[:12],
        "branch": names[0] if names else "（不在任何本地分支上）",
        "dirty": args.base == "HEAD" and bool(git("status", "--porcelain", "--untracked-files=no").strip()),
        "published_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # 要放进提交的文件：{分支里的路径: 本地文件}
    files: dict[str, Path] = {}
    for p in (onnx[0], meta_path):
        files[(MODELS_DIR / p.name).as_posix()] = p
    rel_config = args.config if not args.config.is_absolute() else DEFAULT_CONFIG
    files[Path(rel_config).as_posix()] = config
    for p in sorted(export.iterdir()):
        if p.is_file() and not p.name.endswith(".tmp.npy"):
            files[(workdir / "export" / p.name).as_posix()] = p
    data_files: dict[str, Path] = {}
    if not args.no_data:
        if workdir_abs.exists():
            for p in sorted(workdir_abs.rglob("*")):
                rel = p.relative_to(workdir_abs)
                if p.is_file() and rel.parts[0] not in SKIP_WORK_DIRS and rel.parts[0] != "export":
                    data_files[(workdir / rel).as_posix()] = p
        data_files.update(_data_dir_files(args.repo, data_dirs))
    files.update(data_files)
    notes_text = render_notes(args.version, meta, notes, code, sorted(files) + [NOTES_FILE])

    # 用临时 index 在 base 之上造一个提交：不碰工作区，不切分支
    with tempfile.TemporaryDirectory() as tmp:
        tgit = Git(args.repo, {"GIT_INDEX_FILE": str(Path(tmp) / "index")})
        tgit("read-tree", base)
        info = []
        for path, local in files.items():
            info.append(f"100644 {tgit('hash-object', '-w', str(local)).strip()}\t{path}")
        info.append(f"100644 {tgit('hash-object', '-w', '--stdin', input=notes_text).strip()}\t{NOTES_FILE}")
        tgit("update-index", "--add", "--index-info", input="\n".join(info) + "\n")
        tree = tgit("write-tree").strip()
    extra_dirs = "" if not data_dirs else "，以及 " + "、".join(d.as_posix() for d in data_dirs)
    message = (
        f"唤醒词模型 {args.version}：{meta.get('wake_phrase', '')}\n\n"
        + "\n".join(summary_lines(meta))
        + f"\n\n训练代码 {base[:12]}；本提交带模型、配置与 {workdir.as_posix()} 下的训练数据"
        + f"（不含 resources / features）{extra_dirs}。\n"
    )
    commit = git("commit-tree", tree, "-p", base, "-m", message).strip()
    git("update-ref", f"refs/heads/{branch}", commit)
    print(
        f"created {branch} = {commit[:12]} on top of {base[:12]} "
        f"({len(files) + 1} files, {len(data_files)} of training data)"
    )
    if args.push:
        force = ["--force"] if args.force else []
        git("push", *force, args.remote, f"refs/heads/{branch}:refs/heads/{branch}")
        print(f"pushed to {args.remote}/{branch}")
    else:
        print(f"not pushed; run: git push {args.remote} {branch}")
    print(f"pull it with: python scripts/wakeword_model.py pull {args.version}")
    return 0


def main(argv: list[str] | None = None) -> int:
    def common(defaults: bool) -> argparse.ArgumentParser:
        """--repo / --remote 写在子命令前后都行；子命令那份不带默认值，免得盖掉前面给的。"""
        p = argparse.ArgumentParser(add_help=False)
        sup = argparse.SUPPRESS
        p.add_argument("--repo", type=Path, default=REPO_ROOT if defaults else sup, help="仓库根目录")
        p.add_argument("--remote", default="origin" if defaults else sup, help="git 远端名（默认 origin）")
        return p

    ap = argparse.ArgumentParser(
        prog="scripts/wakeword_model.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[common(True)],
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = common(False)  # type: ignore[assignment]

    p = sub.add_parser("list", help="列出所有模型版本", parents=[common])
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("show", help=f"打印某个版本的 {NOTES_FILE}", parents=[common])
    p.add_argument("version")
    p.set_defaults(fn=cmd_show)

    p = sub.add_parser("pull", help="只把模型文件取到本地包目录", parents=[common])
    p.add_argument("version", nargs="?", help=f"版本号；默认读 {MODELS_DIR}/VERSION")
    p.add_argument("--dest", type=Path, default=MODELS_DIR, help=f"目标目录（默认 {MODELS_DIR}）")
    p.set_defaults(fn=cmd_pull)

    p = sub.add_parser("publish", help="把当前代码 + 训练产物 + 训练数据发布成新版本分支", parents=[common])
    p.add_argument("version", help="新版本号，如 v3")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="训练配置 YAML（决定 workdir）")
    p.add_argument("--from", dest="src", type=Path, help="导出目录（默认 <workdir>/export）")
    p.add_argument("--base", default="HEAD", help="作为父提交的代码提交（默认 HEAD）")
    p.add_argument("--no-data", action="store_true", help="不带训练数据，只带模型与导出产物")
    p.add_argument("--notes", default="", help=f"版本说明（写进 {NOTES_FILE}）")
    p.add_argument("--notes-file", help="从文件读版本说明（Markdown）")
    p.add_argument("--push", action="store_true", help="创建后立刻推到远端")
    p.add_argument("--force", action="store_true", help="版本已存在时覆盖")
    p.set_defaults(fn=cmd_publish)

    args = ap.parse_args(argv)
    args.repo = args.repo.resolve()
    if hasattr(signal, "SIGPIPE"):  # `... | head` 关掉管道时安静退出
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
