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

Les jeux de données sont définis dans `configs/datasets.yaml`. Le script reconnaît
les URL Kaggle, Hugging Face, Google Drive et les URL directes d'archives :

```bash
python scripts/download_dataset.py
```

Pour changer de dataset, modifier uniquement `dataset_name` dans la fonction `main`.

## Lancer les tests

```bash
pytest tests/
```

### Tester les trois degradations

Les tests ci-dessous utilisent `data/degradation_test/original.jpg`, verifient que
la sortie conserve la meme taille et le type `uint8`, puis enregistrent une image
pour inspection visuelle.

```bash
pytest tests/degradation/test_fog.py -s
```

Genere `data/degradation_test/foggy.jpg`. Le test de brouillard charge Depth
Anything la premiere fois et peut donc etre plus long.

```bash
pytest tests/degradation/test_rain.py -s
```

Genere `data/degradation_test/rainy.jpg` avec le profil `torrential`, une longueur
de goutte de `20` et une seed `42`.

```bash
pytest tests/degradation/test_lowlight.py -s
```

Genere `data/degradation_test/lowlight.jpg` avec luminosite `-0.4`, contraste
`-0.1`, saturation `0.8` et seed `42`.

Pour executer les trois tests :

```bash
pytest tests/degradation/test_fog.py tests/degradation/test_rain.py tests/degradation/test_lowlight.py -s
```

Ces tests sont des tests d'image isoles et utilisent leurs propres parametres. Pour
tester les valeurs de `configs/degradation.yaml`, utiliser la synthese ci-dessous.

## Lancer la synthese des degradations

La classification jour/nuit et la synthese sont deux etapes separees. Lancer
d'abord la classification :

```bash
python -m src.preprocessing.day_night_classifier
```

Elle copie les images de jour dans `data/splits/` et les images de nuit dans
`data/real_lowlight_test/`. Lancer ensuite la synthese :

```bash
python -m src.degradation.degradation_pipeline
```

La synthese lit `configs/degradation.yaml` et ecrit les images et annotations dans
`data/synthetic/{fog,rain,lowlight}/`.
Modifier les parametres dans `configs/degradation.yaml` si necessaire. Mais les valeurs présentes dedans sont déjà optimales.


## Regler la faible luminosite en direct

Lancer l’interface Gradio avec l’image de test prechargee (un port libre est choisi automatiquement) :

```bash
python scripts/lowlight_ui.py
```

Les curseurs mettent a jour l’aperçu immediatement. Pour imposer le port `7860`, utiliser
`python scripts/lowlight_ui.py --port 7860`. Pour un acces depuis une autre machine,
ajouter `--host 0.0.0.0`.
