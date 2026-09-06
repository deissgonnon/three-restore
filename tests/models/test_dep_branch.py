"""Tests pour src/models/dep_branch/dep_branch.py"""

import pytest
from src.models.dep_branch.dep_branch import DEPBranch


def test_forward_not_implemented():
    branch = DEPBranch()
    with pytest.raises(NotImplementedError):
        branch.forward(None)
