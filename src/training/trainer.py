"""Boucle d'entraînement du réseau de restauration.

Hyperparamètres dans configs/training.yaml.
"""

import torch

class Trainer:
    """Orchestre l'entraînement : forward, perte, backward, logging."""

    def __init__(self, model, optimizer, loss_fn, device: str = "cuda"):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.model.to(self.device)

    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (degraded, clean) in enumerate(dataloader):
            degraded = degraded.to(self.device)
            clean = clean.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(degraded)
            
            # Primary loss (e.g. L1 + Perceptual)
            main_loss = self.loss_fn(output, clean)
            
            # Auxiliary routing loss for MoCE-IR
            aux_loss = 0.0
            
            # If the model is wrapped in MoceIrBaseline, access the underlying model
            # Otherwise if it's directly MoCEIR, we get total_loss
            if hasattr(self.model, "model") and hasattr(self.model.model, "total_loss"):
                aux_loss = self.model.model.total_loss
            elif hasattr(self.model, "total_loss"):
                aux_loss = self.model.total_loss
            elif isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                if hasattr(self.model.module, "total_loss"):
                    aux_loss = self.model.module.total_loss
                elif hasattr(self.model.module, "model") and hasattr(self.model.module.model, "total_loss"):
                    aux_loss = self.model.module.model.total_loss
                
            loss = main_loss + aux_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}")
            
        return total_loss / len(dataloader)

    @torch.no_grad()
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        
        for batch_idx, (degraded, clean) in enumerate(dataloader):
            degraded = degraded.to(self.device)
            clean = clean.to(self.device)
            
            # Assuming output is the first element if a tuple is returned (though moce_ir returns tensor)
            output = self.model(degraded)
            
            loss = self.loss_fn(output, clean)
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
