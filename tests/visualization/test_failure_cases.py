"""Tests pour src/visualization/failure_cases.py"""

import pytest
from src.visualization.failure_cases import plot_failure_case


def test_plot_failure_case_not_implemented():
    with pytest.raises(NotImplementedError):
        plot_failure_case(image=None, restored=None,
                           detections_before=None, detections_after=None)
