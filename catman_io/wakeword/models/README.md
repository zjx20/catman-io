# 唤醒词模型放哪

这个目录在代码分支里是空的（只有本说明和 `VERSION`）。模型文件（`siu_maau_jan.onnx` + 同名 `.json`）
和训练它的代码、合成数据一起放在 `models/wakeword/<版本>` 分支里，每个版本一条分支：

```bash
python scripts/wakeword_model.py pull          # 取 VERSION 里写的推荐版本到这个目录
python scripts/wakeword_model.py pull v0       # 或指定版本
python scripts/wakeword_model.py list          # 有哪些版本
python scripts/wakeword_model.py show v2       # 某个版本的说明与评估
```

`pull` 只下载模型那两个文件（部分克隆），并在这里写一个 `INSTALLED` 记录版本；这三样都被 gitignore。
训练完要发布新版本：`python scripts/wakeword_model.py publish v3 --notes "..." --push`，
详见 [training/wakeword/README.md](../../../training/wakeword/README.md)。
