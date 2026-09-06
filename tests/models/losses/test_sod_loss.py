"""Tests pour src/models/losses/sod_loss.py"""

import pytest
from src.models.losses.sod_loss import ScaleAwareObjectDetailLoss


def test_forward_not_implemented():
    loss_fn = ScaleAwareObjectDetailLoss()
    with pytest.raises(NotImplementedError):
        loss_fn.forward(None, None)
