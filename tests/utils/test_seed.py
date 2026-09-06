"""Tests pour src/utils/seed.py"""

import pytest
from src.utils.seed import set_seed


def test_set_seed_not_implemented():
    with pytest.raises(NotImplementedError):
        set_seed(42)
