"""Tests pour src/models/losses/common_losses.py"""

import pytest
from src.models.losses.common_losses import PerceptualLoss


def test_forward_not_implemented():
    loss_fn = PerceptualLoss()
    with pytest.raises(NotImplementedError):
        loss_fn.forward(None, None)
