"""Tests pour src/datasets/visdrone_loader.py"""

import pytest
from src.datasets.visdrone_loader import VisDroneDataset


def test_dataset_length_not_implemented():
    dataset = VisDroneDataset(root_dir="dummy", split="train")
    with pytest.raises(NotImplementedError):
        len(dataset)
