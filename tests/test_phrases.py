from training.wakeword import phrases


def test_positive_variants_contain_wake_phrase_at_end():
    for text in phrases.POSITIVE_VARIANTS:
        assert phrases.WAKE_PHRASE in text
        # 正样本右对齐，唤醒词后面只允许标点
        tail = text.split(phrases.WAKE_PHRASE, 1)[1]
        assert all(not ch.isalnum() for ch in tail), text


def test_negatives_never_contain_wake_phrase():
    for text in phrases.all_negatives():
        assert phrases.WAKE_PHRASE not in text, text


def test_no_duplicates():
    for name in (
        "POSITIVE_VARIANTS",
        "ADVERSARIAL",
        "NEAR_HOMOPHONES",
        "GENERAL_CANTONESE",
        "GENERAL_MANDARIN",
        "GENERAL_ENGLISH",
    ):
        items = getattr(phrases, name)
        assert len(items) == len(set(items)), name


def test_near_homophones_are_kept_out_of_training():
    assert not set(phrases.NEAR_HOMOPHONES) & set(phrases.ADVERSARIAL)
    from training.wakeword.config import TrainingConfig
    from training.wakeword.features import gather_samples
    from training.wakeword.tts import ClipRecord, plan_jobs

    cfg = TrainingConfig()
    jobs = plan_jobs(cfg)
    assert {j.label for j in jobs} == {"positive", "adversarial", "homophone", "general"}
    records = [
        ClipRecord(j.text, j.voice, j.rate, j.pitch, j.label, j.lang, f"{j.stem}.wav", 1.0, "train")
        for j in jobs
    ]
    groups = gather_samples(cfg, records)
    assert not any(s.kind == "homophone" for s in groups["negative_train"])
    assert any(s.kind == "homophone" for s in groups["negative_val"])
    assert all(groups[k] == [] for k in groups if k.startswith("extra_"))

    # train_on_homophones=True 时近音短语回到训练集；留出的音色（正负样本）只进验证集
    cfg.data.train_on_homophones = True
    cfg.data.holdout_voices = ["zh-HK-WanLungNeural"]
    groups = gather_samples(cfg, records)
    assert any(s.kind == "homophone" for s in groups["negative_train"])
    for lb in ("positive", "negative"):
        assert not any(s.voice == "zh-HK-WanLungNeural" for s in groups[f"{lb}_train"])
        assert any(s.voice == "zh-HK-WanLungNeural" for s in groups[f"{lb}_val"])


def test_real_recordings_go_to_their_own_groups(tmp_path):
    """真人录音：extra_*_dirs 全部训练、extra_*_val_dirs 只验证，子目录名是说话人，目录不存在要报错。"""
    import pytest

    from training.wakeword.config import TrainingConfig
    from training.wakeword.features import gather_samples

    for sub in ("positive/male", "positive/female", "positive_val/male", "negative"):
        (tmp_path / sub).mkdir(parents=True)
    for p in (
        "positive/male/a.wav",
        "positive/male/b.wav",
        "positive/female/c.wav",
        "positive_val/male/d.wav",
    ):
        (tmp_path / p).write_bytes(b"")
    (tmp_path / "negative/tv.wav").write_bytes(b"")
    cfg = TrainingConfig()
    cfg.data.extra_positive_dirs = [str(tmp_path / "positive")]
    cfg.data.extra_positive_val_dirs = [str(tmp_path / "positive_val")]
    cfg.data.extra_negative_dirs = [str(tmp_path / "negative")]
    cfg.data.extra_positive_weight = 5
    groups = gather_samples(cfg, [])
    train, val = groups["extra_positive_train"], groups["extra_positive_val"]
    assert [(s.path.name, s.voice, s.weight) for s in train] == [
        ("c.wav", "female", 5),
        ("a.wav", "male", 5),
        ("b.wav", "male", 5),
    ]
    assert [(s.path.name, s.voice, s.weight, s.positive) for s in val] == [("d.wav", "male", 1, True)]
    assert all(s.kind == "extra" for s in train + val)
    # 文件直接放在目录下：目录名当说话人
    assert [(s.voice, s.positive, s.weight) for s in groups["extra_negative_train"]] == [
        ("negative", False, 2)
    ]
    assert groups["extra_negative_val"] == []
    assert not any(s.kind == "extra" for s in groups["positive_train"] + groups["positive_val"])

    cfg.data.extra_positive_dirs = [str(tmp_path / "nope")]
    with pytest.raises(SystemExit, match="not found"):
        gather_samples(cfg, [])


def test_plan_jobs_respects_limits_and_split():
    from training.wakeword.config import TrainingConfig
    from training.wakeword.tts import plan_jobs, split_for

    cfg = TrainingConfig()
    cfg.tts.max_positive_clips = 20
    cfg.tts.max_negative_clips = 30
    jobs = plan_jobs(cfg)
    assert sum(j.label == "positive" for j in jobs) == 20
    assert sum(j.label != "positive" for j in jobs) == 30
    assert len({j.stem for j in jobs}) == len(jobs)
    assert split_for("abc", 0.0) == "train" and split_for("abc", 1.0) == "val"
