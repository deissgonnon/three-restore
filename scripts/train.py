#!/usr/bin/env python3
"""Lance l'entraînement du réseau DEP-RNet ou d'une baseline.

Usage: python scripts/train.py --baseline moce_ir
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path

# Fix pour les imports locaux depuis la racine du projet
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Charge les variables d'environnement depuis .env (clé API wandb, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from src.baselines.moce_ir import MoceIrBaseline
from src.datasets.moce_ir_loader import MoCEIRDataset
from src.training.trainer import Trainer
from src.visualization.training_samples import (
    select_validation_samples,
    save_epoch_samples,
)


def load_yaml(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_wandb(training_cfg: dict, args):
    """Initialise un run wandb si activé dans la config. Retourne le run ou None."""
    wb_cfg = training_cfg.get("wandb", {})
    if not wb_cfg.get("enabled", True):
        return None
    try:
        import wandb
    except ImportError:
        print("wandb n'est pas installé — suivi désactivé. Installez-le avec: pip install wandb")
        return None

    # Authentification automatique : clé API depuis la config, sinon depuis la
    # variable d'environnement WANDB_API_KEY (chargée depuis .env ou exportée).
    api_key = wb_cfg.get("api_key") or os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key, relogin=True)
    elif not wandb.api.api_key:
        print("Aucune clé API wandb trouvée. Définissez 'wandb.api_key' dans "
              "configs/training.yaml, la variable WANDB_API_KEY dans .env, "
              "ou lancez 'wandb login'.")

    run_name = wb_cfg.get("name") or f"{args.baseline}_bs{args.batch_size}_lr{args.lr}"
    run = wandb.init(
        project=wb_cfg.get("project", "DEP-RNet"),
        entity=wb_cfg.get("entity"),
        name=run_name,
        config={
            "baseline": args.baseline,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "patch_size": args.patch_size,
            "optimizer": training_cfg.get("optimizer", "adamw"),
            "weight_decay": training_cfg.get("weight_decay", 1e-4),
            "scheduler": training_cfg.get("scheduler", "cosine"),
            "num_workers": training_cfg.get("num_workers", 4),
            "seed": training_cfg.get("seed", 42),
        },
        save_code=wb_cfg.get("save_code", True),
    )
    print(f"wandb run initialisé: {run.name} (https://wandb.ai/{run.entity}/{run.project}/{run.id})")
    return run


def main():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--baseline", type=str, default="moce_ir", help="Nom de la baseline à entraîner")
    parser.add_argument("--epochs", type=int, default=100, help="Nombre d'époques")
    parser.add_argument("--batch_size", type=int, default=4, help="Taille du batch")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--patch_size", type=int, default=128, help="Taille des crops MoCE-IR (128 comme l'officiel)")
    parser.add_argument("--no_wandb", action="store_true", help="Désactive le suivi wandb")
    parser.add_argument("--samples_per_degradation", type=int, default=2,
                        help="Nombre d'images du val set à suivre par dégradation (défaut: 2)")
    parser.add_argument("--no_samples", action="store_true",
                        help="Désactive la sauvegarde des échantillons visuels par époque")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} | GPUs disponibles: {torch.cuda.device_count()}")

    # 1. Charger les configs
    datasets_cfg = load_yaml("configs/datasets.yaml")["datasets"]["visdrone"]
    training_cfg = load_yaml("configs/training.yaml")

    # 2. Préparer les DataLoaders
    splits_dir = datasets_cfg.get("splits_dir", "data/splits")
    synthetic_dirs = datasets_cfg.get("synthetic_dirs", {
        "fog": "data/synthetic/fog",
        "rain": "data/synthetic/rain",
        "lowlight": "data/synthetic/lowlight"
    })

    print("Initializing datasets...")
    # Dataloader dédié MoCE-IR : procédure officielle (fusion des tâches, crop 128,
    # augmentation), retourne (meta, lr, hr).
    train_dataset = MoCEIRDataset(splits_dir, synthetic_dirs, split="train", patch_size=args.patch_size)
    val_dataset = MoCEIRDataset(splits_dir, synthetic_dirs, split="val", patch_size=args.patch_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Sélection des échantillons visuels fixes (2 par dégradation) depuis le val set.
    # Ces images seront dégradées et restaurées à chaque époque pour suivre la progression.
    samples = None
    if not args.no_samples:
        samples = select_validation_samples(
            val_dataset,
            n_per_degradation=args.samples_per_degradation,
            seed=training_cfg.get("seed", 42),
        )
        print(f"{len(samples)} échantillons visuels sélectionnés dans le val set "
              f"({args.samples_per_degradation} par dégradation)")

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

    # 5. Suivi wandb (désactivable via --no_wandb)
    use_wandb = not args.no_wandb
    run = setup_wandb(training_cfg, args) if use_wandb else None
    use_wandb = run is not None

    # 6. Entraînement
    trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn,
                      device=device, use_wandb=use_wandb,
                      log_interval=training_cfg.get("wandb", {}).get("log_interval", 10))

    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")

        # Informe le trainer du nombre d'époques restantes pour l'ETA
        trainer.remaining_epochs = args.epochs - epoch
        trainer.start_epoch()

        train_loss = trainer.train_one_epoch(train_loader, epoch=epoch)
        print(f"Train Loss: {train_loss:.4f}")

        val_loss = trainer.validate(val_loader)
        print(f"Val Loss: {val_loss:.4f}")

        scheduler.step()

        # Estimation du temps restant (ETA)
        eta_seconds = trainer.end_epoch()
        if eta_seconds is not None:
            print(f"ETA: {trainer._format_eta(eta_seconds)} restant")

        # Log des métriques par époque
        if use_wandb:
            import wandb
            log_dict = {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
            if eta_seconds is not None:
                log_dict["eta_seconds"] = eta_seconds
            wandb.log(log_dict)

        # Sauvegarde des échantillons visuels (dégradés + restaurés) à chaque époque
        if samples is not None:
            save_epoch_samples(model, samples, epoch=epoch,
                               output_dir="outputs/figures", device=device)

        # Sauvegarde
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            Path("outputs").mkdir(exist_ok=True)
            # En multi-GPU, on sauvegarde le modèle non-enveloppé
            state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state, f"outputs/best_moce_ir.pth")
            print(f"Model saved to outputs/best_moce_ir.pth")

    if use_wandb:
        import wandb
        wandb.finish()

if __name__ == "__main__":
    main()
