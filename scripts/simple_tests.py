import os
import kagglehub

# --- Authentification (choisis une seule méthode) ---

# Option 1 : variables d'environnement (recommandé sur un serveur)
os.environ["KAGGLE_USERNAME"] = "parfaitbotchi1"
os.environ["KAGGLE_KEY"] = "df0eac84ef1f6ae77ac679a7b170fad3"

# Option 2 (alternative) : interactif
#kagglehub.login()

# --- Upload ---

handle = "parfaitbotchi1/DEP-RNet"
local_dataset_dir = "/workspace/202427000031/DEP-RNet/data"

# Première fois : crée le dataset
kagglehub.dataset_upload(handle, local_dataset_dir)

# Fois suivantes : nouvelle version avec notes
# kagglehub.dataset_upload(handle, local_dataset_dir, version_notes="mise a jour des donnees restaurees")

# Pour ignorer certains fichiers/dossiers
# kagglehub.dataset_upload(
#     handle,
#     local_dataset_dir,
#     ignore_patterns=["*.tmp", "checkpoints/"]
# )