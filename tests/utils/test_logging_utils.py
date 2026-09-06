"""Tests pour src/utils/logging_utils.py"""

import pytest
from src.utils.logging_utils import setup_logger


def test_setup_logger_not_implemented():
    with pytest.raises(NotImplementedError):
        setup_logger("test_logger", "dummy_dir")
