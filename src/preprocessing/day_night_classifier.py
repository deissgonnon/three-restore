"""Classe les images VisDrone en jour et nuit."""

import shutil
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
        device: str | None = None,
    ):
        self.prompt = prompt
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Sam3Processor.from_pretrained(model_name)
        self.model = Sam3Model.from_pretrained(model_name).to(self.device).eval()

    def is_night(self, image: Image.Image) -> bool:
        """Retourne True si l'image est classée comme image de nuit."""
        inputs = self.processor(
            images=image, text=self.prompt, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        score = torch.sigmoid(outputs.presence_logits).item()
        return score >= self.threshold

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

        for image_path in tqdm(image_paths, desc=f"Classification {split_name}"):
            with Image.open(image_path) as image:
                destination = (
                    night_dir / split_name
                    if self.is_night(image)
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
