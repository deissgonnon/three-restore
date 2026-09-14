"""Boucle d'entraînement du réseau de restauration.

Hyperparamètres dans configs/training.yaml.
"""

import torch


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

    def train_one_epoch(self, dataloader, epoch: int = 0):
        self.model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
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

            if batch_idx % self.log_interval == 0:
                msg = f"Epoch {epoch} Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}"
                print(msg)
                if self.use_wandb:
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

        return total_loss / len(dataloader)

    @torch.no_grad()
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            degraded, clean = self._unpack(batch)
            degraded = degraded.to(self.device)
            clean = clean.to(self.device)

            output = self.model(degraded)

            loss = self.loss_fn(output, clean)
            total_loss += loss.item()

        return total_loss / len(dataloader)
