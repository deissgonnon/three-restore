"""Tests pour src/datasets/split_manager.py"""

import pytest
from src.datasets.split_manager import load_split, create_splits


def test_load_split_not_implemented():
    with pytest.raises(NotImplementedError):
        load_split("train", "dummy_dir")


def test_create_splits_not_implemented():
    with pytest.raises(NotImplementedError):
        create_splits([], {"train": 0.7, "val": 0.15, "test": 0.15}, 42, "dummy_dir")
