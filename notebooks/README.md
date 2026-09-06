# Notebooks exploratoires

`visualize-visdrone-dataset.ipynb` : travail préparatoire (classification
jour/nuit via SAM3, prototypage de la synthèse de brouillard par modèle
de Koschmieder + bruit fractal). La logique utile a été migrée vers :

- `src/preprocessing/day_night_classifier.py`
- `src/degradation/fog.py` + `src/degradation/noise.py`

