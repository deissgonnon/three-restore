#!/usr/bin/env python3
"""Lance l'entraînement du réseau DEP-RNet ou d'une baseline.

Usage: python scripts/train.py --baseline moce_ir
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path

# Fix pour les imports locaux depuis la racine du projet
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.baselines.moce_ir import MoceIrBaseline
from src.datasets.mixed_degradation_loader import MixedDegradationDataset
from src.training.trainer import Trainer


def load_yaml(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--baseline", type=str, default="moce_ir", help="Nom de la baseline à entraîner")
    parser.add_argument("--epochs", type=int, default=100, help="Nombre d'époques")
    parser.add_argument("--batch_size", type=int, default=4, help="Taille du batch")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Charger les configs
    datasets_cfg = load_yaml("configs/datasets.yaml")["datasets"]["visdrone"]
    
    # 2. Préparer les DataLoaders
    splits_dir = datasets_cfg.get("splits_dir", "data/splits")
    synthetic_dirs = datasets_cfg.get("synthetic_dirs", {
        "fog": "data/synthetic/fog",
        "rain": "data/synthetic/rain",
        "lowlight": "data/synthetic/lowlight"
    })
    
    print("Initializing datasets...")
    train_dataset = MixedDegradationDataset(splits_dir, synthetic_dirs, split="train")
    val_dataset = MixedDegradationDataset(splits_dir, synthetic_dirs, split="val")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 3. Initialiser le modèle
    if args.baseline == "moce_ir":
        print("Initializing MoCE-IR model...")
        baselines_cfg = load_yaml("configs/baselines.yaml")["baselines"]["all_in_one"]["moce_ir"]
        # Enlever les clés qui ne sont pas pour le constructeur de MoCEIR
        kwargs = {k: v for k, v in baselines_cfg.items() if k not in ["weights_path"]}
        
        wrapper = MoceIrBaseline(**kwargs)
        # On extrait le modèle PyTorch interne pour l'entraînement (le wrapper est conçu pour l'inférence)
        model = wrapper.model
    else:
        raise ValueError(f"Baseline non supportée: {args.baseline}")
    
    # 4. Fonction de perte et Optimiseur
    loss_fn = nn.L1Loss()  # La configuration de perte (loss.yaml) peut être intégrée ici plus tard
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 5. Entraînement
    trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn, device=device)
    
    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        
        train_loss = trainer.train_one_epoch(train_loader)
        print(f"Train Loss: {train_loss:.4f}")
        
        val_loss = trainer.validate(val_loader)
        print(f"Val Loss: {val_loss:.4f}")
        
        scheduler.step()
        
        # Sauvegarde
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            Path("outputs").mkdir(exist_ok=True)
            torch.save(model.state_dict(), f"outputs/best_moce_ir.pth")
            print(f"Model saved to outputs/best_moce_ir.pth")

if __name__ == "__main__":
    main()
