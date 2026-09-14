"""Boucle d'entraînement du réseau de restauration.

Hyperparamètres dans configs/training.yaml.
"""

import time

import torch
from tqdm import tqdm


class Trainer:
    """Orchestre l'entraînement : forward, perte, backward, logging.

    Gère le multi-GPU via ``torch.nn.DataParallel`` et le suivi des métriques
    via Weights & Biases (wandb) si un run est actif.
    """

    def __init__(self, model, optimizer, loss_fn, device: str = "cuda",
                 use_wandb: bool = True, log_interval: int = 10):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.use_wandb = use_wandb
        self.log_interval = log_interval

        # Multi-GPU : enveloppe le modèle dans DataParallel si plusieurs GPU dispo.
        self.multi_gpu = False
        if device.startswith("cuda") and torch.cuda.device_count() > 1:
            self.multi_gpu = True
            print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
            self.model = torch.nn.DataParallel(self.model)

        self.model.to(self.device)

        # Suivi du temps pour l'estimation du temps restant (ETA).
        self.epoch_times = []          # durée de chaque époque (train + val)
        self._epoch_start = None       # timestamp de début de l'époque courante

    @staticmethod
    def _unpack(batch):
        """Accepte (meta, degraded, clean) (MoCE-IR officiel) ou (degraded, clean)."""
        if len(batch) == 3:
            _, degraded, clean = batch
        else:
            degraded, clean = batch
        return degraded, clean

    def _get_aux_loss(self):
        """Récupère la perte auxiliaire de routage MoCE-IR (total_loss)."""
        model = self.model
        if self.multi_gpu:
            model = self.model.module
        if hasattr(model, "total_loss"):
            return model.total_loss
        if hasattr(model, "model") and hasattr(model.model, "total_loss"):
            return model.model.total_loss
        return 0.0

    @staticmethod
    def _format_eta(seconds: float) -> str:
        """Formate une durée en secondes en 'Xh Ym Zs'."""
        seconds = max(0, int(seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h}h {m:02d}m {s:02d}s"
        if m > 0:
            return f"{m}m {s:02d}s"
        return f"{s}s"

    def start_epoch(self):
        """Marque le début d'une époque pour mesurer sa durée."""
        self._epoch_start = time.time()

    def end_epoch(self):
        """Enregistre la durée de l'époque terminée et retourne l'ETA restant.

        Retourne le temps restant estimé (en secondes) pour la fin de
        l'entraînement, ou None si pas assez d'époques pour estimer.
        """
        if self._epoch_start is None:
            return None
        elapsed = time.time() - self._epoch_start
        self.epoch_times.append(elapsed)
        self._epoch_start = None
        return self.eta()

    def eta(self) -> float | None:
        """Estime le temps restant (s) en supposant un rythme constant par époque."""
        if not self.epoch_times:
            return None
        avg_epoch = sum(self.epoch_times) / len(self.epoch_times)
        return avg_epoch * self.remaining_epochs

    @property
    def remaining_epochs(self) -> int:
        """Nombre d'époques restantes (mis à jour par le script d'entraînement)."""
        return getattr(self, "_remaining_epochs", 0)

    @remaining_epochs.setter
    def remaining_epochs(self, value: int):
        self._remaining_epochs = value

    def train_one_epoch(self, dataloader, epoch: int = 0):
        self.model.train()
        total_loss = 0.0

        # Barre de progression par époque (format tqdm : Epoch 3: 45%|██▌| 450/1000 [02:15<02:45, 3.33it/s])
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", unit="it",
                    dynamic_ncols=True, leave=False)
        for batch_idx, batch in enumerate(pbar):
            degraded, clean = self._unpack(batch)
            degraded = degraded.to(self.device)
            clean = clean.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(degraded)

            # Primary loss (e.g. L1 + Perceptual)
            main_loss = self.loss_fn(output, clean)

            # Auxiliary routing loss for MoCE-IR
            aux_loss = self._get_aux_loss()

            # Poids balance_loss_weight officiel (options.py) = 0.01
            loss = main_loss + 0.01 * aux_loss

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Mise à jour de la barre avec la loss courante
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if batch_idx % self.log_interval == 0 and self.use_wandb:
                try:
                    import wandb
                    wandb.log({
                        "train/batch_loss": loss.item(),
                        "train/batch_main_loss": main_loss.item(),
                        "train/batch_aux_loss": aux_loss.item() if torch.is_tensor(aux_loss) else aux_loss,
                        "train/batch": epoch * len(dataloader) + batch_idx,
                    })
                except ImportError:
                    pass

        pbar.close()
        return total_loss / len(dataloader)

    @torch.no_grad()
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0

        pbar = tqdm(dataloader, desc="Val", unit="it",
                    dynamic_ncols=True, leave=False)
        for batch_idx, batch in enumerate(pbar):
            degraded, clean = self._unpack(batch)
            degraded = degraded.to(self.device)
            clean = clean.to(self.device)

            output = self.model(degraded)

            loss = self.loss_fn(output, clean)
            total_loss += loss.item()

        pbar.close()
        return total_loss / len(dataloader)
