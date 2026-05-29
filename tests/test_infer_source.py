from pathlib import Path

from poker_yolo.infer_source import collect_images_under, resolve_infer_source


def test_resolve_infer_source_expands_dataset_root(tmp_path: Path) -> None:
    hands = tmp_path / "test" / "images"
    hands.mkdir(parents=True)
    (hands / "a.jpg").write_bytes(b"x")
    kaggle_train = tmp_path / "kaggle" / "train" / "images"
    kaggle_train.mkdir(parents=True)
    (kaggle_train / "skip.jpg").write_bytes(b"x")
    kaggle_test = tmp_path / "kaggle" / "test" / "images"
    kaggle_test.mkdir(parents=True)
    (kaggle_test / "keep.jpg").write_bytes(b"x")

    resolved = resolve_infer_source(tmp_path)
    assert isinstance(resolved, list)
    assert len(resolved) == 2
    assert any(p.endswith("a.jpg") for p in resolved)
    assert any(p.endswith("keep.jpg") for p in resolved)
    assert not any("skip.jpg" in p for p in resolved)


def test_resolve_infer_source_keeps_direct_image_dir(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "one.png").write_bytes(b"x")

    resolved = resolve_infer_source(images)
    assert resolved == images


def test_collect_images_under_empty(tmp_path: Path) -> None:
    assert collect_images_under(tmp_path) == []
