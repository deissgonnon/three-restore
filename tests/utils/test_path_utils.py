"""Tests pour src/utils/path_utils.py"""

import os
from src.utils.path_utils import load_paths_config


def test_load_paths_config_reads_yaml(tmp_path):
    config_file = tmp_path / "paths.yaml"
    config_file.write_text("data:\n  raw_visdrone: dummy\n")
    result = load_paths_config(str(config_file))
    assert result["data"]["raw_visdrone"] == "dummy"
