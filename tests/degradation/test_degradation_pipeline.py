"""Tests pour src/degradation/degradation_pipeline.py"""

import pytest
from src.degradation.degradation_pipeline import generate_all_degradations


def test_generate_all_degradations_not_implemented():
    with pytest.raises(NotImplementedError):
        generate_all_degradations(image=None, config={})
