"""Tests pour src/training/checkpoint.py"""

import pytest
from src.training.checkpoint import save_checkpoint, load_checkpoint


def test_save_checkpoint_not_implemented():
    with pytest.raises(NotImplementedError):
        save_checkpoint(model=None, optimizer=None, epoch=1, checkpoint_dir="dummy")


def test_load_checkpoint_not_implemented():
    with pytest.raises(NotImplementedError):
        load_checkpoint(model=None, optimizer=None, checkpoint_path="dummy.pt")
