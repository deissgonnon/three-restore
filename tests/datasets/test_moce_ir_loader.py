"""Tests pour src/datasets/moce_ir_loader.py"""

import numpy as np
import pytest
import torch
from PIL import Image

from src.datasets.moce_ir_loader import MoCEIRDataset, _data_augmentation

PATCH = 128


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((256, 256, 3), value, dtype=np.uint8)).save(path)


@pytest.fixture
def dataset_dirs(tmp_path):
    """Structure mini : 2 claires en train (dont 1 sans dégradée), 1 claire en val."""
    splits = tmp_path / "splits"
    synthetic = {
        "fog": tmp_path / "synthetic" / "fog",
        "rain": tmp_path / "synthetic" / "rain",
        "lowlight": tmp_path / "synthetic" / "lowlight",
    }
    # Train : a.png a 3 versions dégradées, b.png n'en a aucune (paire ignorée)
    _save(splits / "train" / "images" / "a.png", 200)
    _save(splits / "train" / "images" / "b.png", 200)
    _save(synthetic["fog"] / "train" / "images" / "a.png", 100)
    _save(synthetic["rain"] / "train" / "images" / "a.png", 150)
    _save(synthetic["lowlight"] / "train" / "images" / "a.png", 50)
    # Val : c.png avec seulement fog
    _save(splits / "val" / "images" / "c.png", 200)
    _save(synthetic["fog"] / "val" / "images" / "c.png", 100)
    return splits, synthetic


def test_len_fusion_des_taches(dataset_dirs):
    splits, synthetic = dataset_dirs
    ds = MoCEIRDataset(splits, synthetic, split="train", patch_size=PATCH)
    # 3 tâches x 1 image (b.png ignorée : pas de version dégradée)
    assert len(ds) == 3


def test_getitem_format_officiel(dataset_dirs):
    splits, synthetic = dataset_dirs
    ds = MoCEIRDataset(splits, synthetic, split="train", patch_size=PATCH)
    meta, lr, hr = ds[0]
    # Format officiel : ([chemin, de_id], lr, hr)
    assert isinstance(meta, list) and len(meta) == 2
    assert isinstance(meta[0], str) and isinstance(meta[1], int)
    assert tuple(lr.shape) == (3, PATCH, PATCH)
    assert tuple(hr.shape) == (3, PATCH, PATCH)


def test_paires_degradee_claire_correspondent(dataset_dirs):
    """fog (100) = claire (200) / 2 : la relation tient à travers crop et augmentation,
    ce qui vérifie le couplage lr/hr (même crop, même transformation)."""
    splits, synthetic = dataset_dirs
    ds = MoCEIRDataset(splits, synthetic, split="train", patch_size=PATCH)
    # Tâches triées alphabétiquement : fog=0, lowlight=1, rain=2
    assert ds.pairs[0]["de_type"] == 0
    meta, lr, hr = ds[0]
    assert torch.all(2 * lr == hr)


def test_val_deterministe(dataset_dirs):
    splits, synthetic = dataset_dirs
    ds = MoCEIRDataset(splits, synthetic, split="val", patch_size=PATCH)
    meta_1, lr_1, hr_1 = ds[0]
    meta_2, lr_2, hr_2 = ds[0]
    assert torch.equal(lr_1, lr_2) and torch.equal(hr_1, hr_2)
    assert tuple(lr_1.shape) == (3, PATCH, PATCH)


def test_aucune_paire_leve_erreur(tmp_path):
    _save(tmp_path / "splits" / "train" / "images" / "a.png", 200)
    with pytest.raises(ValueError):
        MoCEIRDataset(tmp_path / "splits", {"fog": tmp_path / "vide"}, split="train")


def test_data_augmentation_modes():
    a = np.arange(24).reshape(2, 3, 4)
    assert np.array_equal(_data_augmentation(a, 0), a)
    assert np.array_equal(_data_augmentation(a, 1), np.flipud(a))
    assert np.array_equal(_data_augmentation(a, 2), np.rot90(a))
    assert np.array_equal(_data_augmentation(a, 3), np.flipud(np.rot90(a)))
    with pytest.raises(ValueError):
        _data_augmentation(a, 8)
