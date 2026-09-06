"""Tests pour src/models/restoration/dep_rnet.py"""

import pytest
from src.models.restoration.dep_rnet import DEPRNet


def test_forward_not_implemented():
    model = DEPRNet()
    with pytest.raises(NotImplementedError):
        model.forward(None)
