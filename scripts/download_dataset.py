#!/usr/bin/env python3
"""Télécharge et extrait un jeu de données depuis une URL."""

import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import yaml


def load_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def extract_archive(archive_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(output_dir)
        return

    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as archive:
            archive.extractall(output_dir, filter="data")
        return

    raise ValueError(f"Format d'archive non supporté : {archive_path}")


def download_dataset(url: str, download_dir: Path) -> Path | None:
    parsed_url = urllib.parse.urlparse(url)
    host = parsed_url.netloc.lower()
    parts = [part for part in parsed_url.path.split("/") if part]

    if host.endswith("kaggle.com") and "datasets" in parts:
        try:
            import kagglehub
        except ImportError as error:
            raise SystemExit("Installez kagglehub pour télécharger depuis Kaggle.") from error

        dataset_index = parts.index("datasets")
        dataset_id = "/".join(parts[dataset_index + 1 : dataset_index + 3])
        if dataset_id.count("/") != 1:
            raise ValueError("URL Kaggle attendue : https://www.kaggle.com/datasets/<owner>/<dataset>")
        print(f"Téléchargement Kaggle : {dataset_id}")
        return Path(kagglehub.dataset_download(dataset_id))

    if host in {"huggingface.co", "hf.co"} and parts and parts[0] == "datasets":
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise SystemExit("Installez huggingface_hub pour télécharger depuis Hugging Face.") from error

        repo_id = "/".join(parts[1:3])
        if repo_id.count("/") != 1:
            raise ValueError(
                "URL Hugging Face attendue : https://huggingface.co/datasets/<owner>/<dataset>"
            )
        print(f"Téléchargement Hugging Face : {repo_id}")
        return Path(
            snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=download_dir)
        )

    if host == "drive.google.com":
        try:
            import gdown
        except ImportError as error:
            raise SystemExit("Installez gdown pour télécharger depuis Google Drive.") from error

        if "folders" in parts:
            print(f"Téléchargement du dossier Google Drive : {url}")
            gdown.download_folder(url, output=str(download_dir), quiet=False)
            return None
        archive_path = download_dir / "google-drive-download"
        print(f"Téléchargement Google Drive : {url}")
        gdown.download(url, output=str(archive_path), fuzzy=True, quiet=False)
        return archive_path

    archive_path = download_dir / "dataset-download"
    print(f"Téléchargement : {url}")
    urllib.request.urlretrieve(url, archive_path)
    return archive_path


def main() -> None:
    dataset_name = "visdrone"
    datasets = load_config(Path("configs/datasets.yaml")).get("datasets", {})
    if dataset_name not in datasets:
        available = ", ".join(sorted(datasets)) or "aucun"
        raise SystemExit(f"Dataset inconnu : {dataset_name}. Disponibles : {available}")

    dataset_config = datasets[dataset_name]
    url = dataset_config.get("url")
    if not url:
        raise SystemExit(f"Aucune URL configurée pour {dataset_name}.")
    output_dir = Path(dataset_config["output_dir"])

    with tempfile.TemporaryDirectory() as temporary_dir:
        download_dir = Path(temporary_dir)
        source = download_dataset(url, download_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if source is None or source.is_dir():
            source_dir = source or download_dir
            print(f"Copie : {source_dir} -> {output_dir}")
            shutil.copytree(source_dir, output_dir, dirs_exist_ok=True)
        else:
            print(f"Extraction : {source} -> {output_dir}")
            extract_archive(source, output_dir)

    print(f"Dataset {dataset_name} disponible dans : {output_dir}")


if __name__ == "__main__":
    main()