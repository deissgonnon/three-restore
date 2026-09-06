"""Tests pour src/datasets/category_filter.py"""

import pytest
from src.datasets.category_filter import filter_annotations_by_category


def test_filter_not_implemented():
    with pytest.raises(NotImplementedError):
        filter_annotations_by_category([], ["car"])
