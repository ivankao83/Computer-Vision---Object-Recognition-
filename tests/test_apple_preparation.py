import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "prepare_apple_dataset",
    Path(__file__).resolve().parents[1] / "scripts" / "prepare_apple_dataset.py",
)
preparation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)


def test_augmented_versions_cannot_cross_splits():
    images = {
        "train": [Path("a.rf.train.jpg"), Path("b.rf.train.jpg"), Path("c.rf.train.jpg")],
        "valid": [Path("a.rf.valid.jpg"), Path("b.rf.valid.jpg")],
        "test": [Path("a.rf.test.jpg")],
    }
    selected = preparation.select_splits(images)
    assert selected == {
        "train": [Path("c.rf.train.jpg")],
        "valid": [Path("b.rf.valid.jpg")],
        "test": [Path("a.rf.test.jpg")],
    }
    assert len(images["train"]) == 3


def test_single_class_conversion_keeps_boxes():
    assert preparation.merge_labels("2 0.5 0.6 0.2 0.3\n1 0.1 0.2 0.1 0.1") == (
        "0 0.5 0.6 0.2 0.3\n0 0.1 0.2 0.1 0.1\n"
    )
    assert preparation.merge_labels("") == ""


def test_unknown_class_is_not_silently_merged():
    with pytest.raises(ValueError, match="Unexpected label"):
        preparation.merge_labels("3 0.5 0.5 0.1 0.1")
