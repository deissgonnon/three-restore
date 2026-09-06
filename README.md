# DEP-RNet — Restauration d'images de drone préservant les preuves de détection

Réseau de restauration d'images VisDrone en conditions météo défavorables
(brouillard, pluie, faible luminosité) qui préserve les indices de détection
des véhicules et des piétons. Voir `docs/suivi_projet.md` pour le plan détaillé.

## Structure du projet

```
dep-rnet/
├── configs/            # Toute valeur/paramètre modifiable (YAML), rien en dur dans le code
├── data/               # Données brutes et générées (non versionnées, sauf splits/)
│   ├── raw/visdrone/           # Dataset VisDrone d'origine (jamais modifié)
│   ├── synthetic/{fog,rain,lowlight}/  # Images dégradées synthétiques
│   ├── splits/                 # Listes train/val/test figées (versionnées)
│   └── real_lowlight_test/     # Images réelles nuit/faible lum. — test de robustesse uniquement
├── src/
│   ├── datasets/       # Chargement VisDrone, filtrage par catégorie, gestion des splits
│   ├── preprocessing/  # Classification jour/nuit (SAM3) et autres filtrages pré-entraînement
│   ├── degradation/    # Synthèse brouillard (Koschmieder+bruit fractal), pluie, faible luminosité
│   ├── models/
│   │   ├── restoration/   # Réseau de restauration principal (DEP-RNet)
│   │   ├── dep_branch/    # Branche allégée de préservation des preuves de détection
│   │   └── losses/        # Perte SOD (Scale-aware Object Detail) + pertes classiques
│   ├── detection/      # Wrapper détecteur fixe (YOLO) + calcul des métriques mAP
│   ├── training/       # Boucle d'entraînement, checkpointing
│   ├── evaluation/     # Métriques de restauration (PSNR/SSIM/LPIPS) + évaluation conjointe
│   ├── baselines/      # Wrappers pour AirNet, PromptIR, MoCE-IR, Restormer, etc.
│   ├── visualization/  # Figures : exemples de dégradation, cas d'échec, comparaisons visuelles
│   └── utils/          # Seeds, logging, gestion des chemins
├── scripts/            # Points d'entrée exécutables (train.py, evaluate_baseline.py, ...)
├── tests/              # Miroir de src/, un fichier de test par module (pytest)
├── experiments/        # Un sous-dossier par run, avec config figée + résultats
├── outputs/            # Artefacts finaux (checkpoints, figures exportées)
├── logs/               # Logs bruts d'entraînement/évaluation
├── notebooks/          # Notebooks exploratoires (travail préparatoire avant intégration à src/)
└── docs/               # Document de suivi, journal de décisions
```

## Règles de base

- **Aucune valeur en dur dans le code** : tout paramètre modifiable vit dans `configs/`.
- **`data/raw/visdrone/` n'est jamais modifié** : les dégradations et filtrages génèrent
  de nouveaux fichiers ailleurs.
- **Les splits train/val/test sont figés** une fois définis dans `data/splits/`.
- **Chaque module de `src/` a son miroir dans `tests/`.**
- Les images réelles de nuit/faible luminosité ne rentrent jamais dans le benchmark
  principal — elles servent uniquement de test de robustesse (voir `docs/suivi_projet.md`).

## Installation

```bash
pip install -r requirements.txt
```

## Télécharger un jeu de données

Les sources sont déclarées dans `configs/datasets.yaml`. Pour télécharger VisDrone,
renseigner son URL puis lancer :

```bash
python scripts/download_dataset.py --dataset visdrone
```

Une URL ponctuelle et un dossier de sortie peuvent aussi être fournis sans modifier
la configuration :

```bash
python scripts/download_dataset.py --dataset visdrone \
  --url "URL_DE_L_ARCHIVE" --output-dir data/raw/visdrone
```

## Lancer les tests

```bash
pytest tests/
```
