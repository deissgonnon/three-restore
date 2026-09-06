"""Tests pour src/training/trainer.py"""

import pytest
from src.training.trainer import Trainer


def test_train_one_epoch_not_implemented():
    trainer = Trainer(model=None, optimizer=None, loss_fn=None, config={})
    with pytest.raises(NotImplementedError):
        trainer.train_one_epoch(dataloader=None)


def test_validate_not_implemented():
    trainer = Trainer(model=None, optimizer=None, loss_fn=None, config={})
    with pytest.raises(NotImplementedError):
        trainer.validate(dataloader=None)
