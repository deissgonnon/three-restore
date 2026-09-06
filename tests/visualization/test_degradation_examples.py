"""Tests pour src/visualization/degradation_examples.py"""

import pytest
from src.visualization.degradation_examples import plot_degradation_examples


def test_plot_degradation_examples_not_implemented():
    with pytest.raises(NotImplementedError):
        plot_degradation_examples(image_pairs=[], degradation_type="fog")
