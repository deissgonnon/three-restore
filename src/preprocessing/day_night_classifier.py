"""Classe les images VisDrone en jour et nuit."""

import shutil
import os
from pathlib import Path

import torch
import yaml
from PIL import Image
from transformers import Sam3Model, Sam3Processor
from tqdm import tqdm


class DayNightClassifier:
    """Classe les images et conserve leurs annotations associées."""

    def __init__(
        self,
        model_name: str = "facebook/sam3",
        prompt: str = "photo taken at night",
        threshold: float = 0.3,
        batch_size: int = 1,
        cache_dir: str | None = None,
        device: str | None = None,
    ):
        self.prompt = prompt
        self.threshold = threshold
        self.batch_size = max(1, batch_size)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        load_options = {"cache_dir": cache_dir, "token": token}
        try:
            self.processor = Sam3Processor.from_pretrained(model_name, **load_options)
            self.model = Sam3Model.from_pretrained(model_name, **load_options).to(self.device).eval()
        except OSError as error:
            if "gated" in str(error).lower() or "401" in str(error):
                raise RuntimeError(
                    f"Le modele Hugging Face {model_name} est protege. "
                    "Accepter son acces sur Hugging Face puis definir HF_TOKEN "
                    "(ou HUGGINGFACE_HUB_TOKEN) avant de relancer."
                ) from error
            raise

    def is_night(self, image: Image.Image) -> bool:
        """Retourne True si l'image est classée comme image de nuit."""
        return self.is_night_batch([image])[0]

    def is_night_batch(self, images: list[Image.Image]) -> list[bool]:
        """Classifie plusieurs images en une seule inférence SAM3."""
        if not images:
            return []

        inputs = self.processor(
            images=images,
            text=[self.prompt] * len(images),
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.presence_logits.reshape(len(images), -1)[:, 0]
        scores = torch.sigmoid(logits)
        return (scores >= self.threshold).tolist()

    def _copy_image_and_annotation(
        self,
        image_path: Path,
        images_dir: Path,
        annotations_dir: Path,
        destination: Path,
    ) -> None:
        relative_path = image_path.relative_to(images_dir)
        output_image = destination / "images" / relative_path
        output_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, output_image)

        annotation = annotations_dir / relative_path.with_suffix(".txt")
        if annotation.exists():
            output_annotation = destination / "annotations" / relative_path.with_suffix(".txt")
            output_annotation.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(annotation, output_annotation)

    def process_split(
        self,
        split_dir: Path,
        split_name: str,
        day_dir: Path,
        night_dir: Path,
    ) -> None:
        """Classe un split et copie chaque annotation avec son image."""
        images_dir = split_dir / "images"
        annotations_dir = split_dir / "annotations"
        image_paths = sorted(
            path
            for path in images_dir.rglob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        for start in tqdm(
            range(0, len(image_paths), self.batch_size),
            desc=f"Classification {split_name}",
        ):
            batch_paths = image_paths[start : start + self.batch_size]
            images = []
            for image_path in batch_paths:
                with Image.open(image_path) as image:
                    images.append(image.convert("RGB"))
            night_flags = self.is_night_batch(images)
            for image_path, is_night in zip(batch_paths, night_flags):
                destination = (
                    night_dir / split_name
                    if is_night
                    else day_dir / split_name
                )
                self._copy_image_and_annotation(
                    image_path, images_dir, annotations_dir, destination
                )

        print(f"{split_name}: {len(image_paths)} images classées")


def main() -> None:
    dataset_name = "visdrone"
    with Path("configs/datasets.yaml").open(encoding="utf-8") as config_file:
        dataset = yaml.safe_load(config_file)["datasets"][dataset_name]
    with Path("configs/preprocessing.yaml").open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)["day_night_classifier"]

    if not config.get("enable", False):
        print("Classification jour/nuit desactivee dans configs/preprocessing.yaml")
        return

    classifier = DayNightClassifier(
        model_name=config.get("model_name", "facebook/sam3"),
        prompt=config.get("prompt", "photo taken at night"),
        threshold=config.get("presence_threshold", 0.3),
        batch_size=config.get("batch_size", 1),
        cache_dir=config.get("cache_dir"),
    )
    for split_name in ("train", "val", "test"):
        classifier.process_split(
            split_dir=Path(dataset[f"{split_name}_dir"]),
            split_name=split_name,
            day_dir=Path("data/splits"),
            night_dir=Path("data/real_lowlight_test"),
        )


if __name__ == "__main__":
    main()
