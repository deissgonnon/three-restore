#!/usr/bin/env python3
"""Télécharge et extrait un jeu de données déclaré dans la configuration."""

import argparse
import hashlib
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import yaml


def load_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Téléchargement : {url}")
    urllib.request.urlretrieve(url, destination)


def verify_checksum(file_path: Path, expected: str | None) -> None:
    if not expected:
        return

    digest = hashlib.sha256()
    with file_path.open("rb") as archive:
        for block in iter(lambda: archive.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 invalide pour {file_path}: {actual}")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Nom déclaré dans configs/datasets.yaml")
    parser.add_argument("--config", type=Path, default=Path("configs/datasets.yaml"))
    parser.add_argument("--url", help="Remplace l'URL de la configuration")
    parser.add_argument("--output-dir", type=Path, help="Remplace le dossier d'extraction")
    parser.add_argument("--archive", type=Path, help="Chemin local d'une archive déjà téléchargée")
    parser.add_argument("--keep-archive", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = load_config(args.config).get("datasets", {})
    if args.dataset not in datasets:
        available = ", ".join(sorted(datasets)) or "aucun"
        raise SystemExit(f"Dataset inconnu : {args.dataset}. Disponibles : {available}")

    dataset_config = datasets[args.dataset]
    output_dir = args.output_dir or Path(dataset_config["output_dir"])
    archive_path = args.archive
    downloaded = archive_path is None

    if archive_path is None:
        url = args.url or dataset_config.get("url")
        if not url:
            raise SystemExit(
                f"Aucune URL pour {args.dataset}. Ajoutez-la dans {args.config} ou utilisez --url."
            )
        archive_path = Path(dataset_config.get("archive_name", f"{args.dataset}.archive"))
        download_file(url, archive_path)

    verify_checksum(archive_path, dataset_config.get("sha256"))
    print(f"Extraction : {archive_path} -> {output_dir}")
    extract_archive(archive_path, output_dir)

    if downloaded and not args.keep_archive:
        archive_path.unlink()
    print(f"Dataset disponible dans : {output_dir}")


if __name__ == "__main__":
    main()