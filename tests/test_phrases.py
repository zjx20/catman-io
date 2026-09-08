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
        "GENERAL_CANTONESE",
        "GENERAL_MANDARIN",
        "GENERAL_ENGLISH",
    ):
        items = getattr(phrases, name)
        assert len(items) == len(set(items)), name


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
