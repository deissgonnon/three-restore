"""Classification jour/nuit via SAM3 (facebook/sam3 + vision-language)."""

import numpy as np
import torch
from PIL import Image
from transformers import Sam3Processor, Sam3Model


class DayNightClassifier:
    """Classifie images en jour/nuit via SAM3 presence score."""

    def __init__(self, model_name: str = "facebook/sam3", prompt: str = "photo taken at night", 
                 threshold: float = 0.3, device: str = None):
        self.model_name = model_name
        self.prompt = prompt
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None

    def _load_model(self):
        if self.processor is None:
            self.processor = Sam3Processor.from_pretrained(self.model_name)
            self.model = Sam3Model.from_pretrained(self.model_name).to(self.device).eval()

    def classify(self, image: Image.Image) -> dict:
        """Classify single image. Returns dict with presence_score and label."""
        self._load_model()
        
        inputs = self.processor(images=image, text=self.prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            presence_score = torch.sigmoid(outputs.presence_logits).item()
        
        label = "night" if presence_score >= self.threshold else "day"
        return {
            "presence_score": presence_score,
            "label": label,
            "prompt": self.prompt
        }

    def classify_batch(self, image_paths: list, batch_size: int = 8) -> list:
        """Classify multiple images."""
        results = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            for path in batch_paths:
                img = Image.open(path).convert("RGB")
                result = self.classify(img)
                result["path"] = path
                results.append(result)
        return results


def classify_day_night(scores, threshold: float = DEFAULT_THRESHOLD):
    """Labellise chaque image 'jour' ou 'nuit' selon le seuil de score."""
    raise NotImplementedError
