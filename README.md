# PLATE AI PRO — Reconnaissance Automatique de Plaques d’Immatriculation

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![EasyOCR](https://img.shields.io/badge/OCR-EasyOCR-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-orange)

## 📌 Description du projet

**PLATE AI PRO** est une application de reconnaissance automatique de plaques d’immatriculation, aussi appelée **ALPR / ANPR** (*Automatic License Plate Recognition*).  
Le système permet de détecter une plaque dans une image ou une vidéo, de l’isoler automatiquement, puis d’extraire son contenu textuel à l’aide de l’OCR.

Le projet repose sur deux grandes briques de Deep Learning :

- **YOLOv8** pour la détection et la localisation des plaques d’immatriculation.
- **EasyOCR** pour la reconnaissance optique des caractères présents sur la plaque.

L’objectif est de fournir une solution rapide, précise et utilisable dans des cas réels comme la gestion de parkings, la surveillance routière, le contrôle d’accès ou l’analyse de trafic.

---

## 🎯 Objectifs

- Détecter automatiquement les plaques d’immatriculation dans des images.
- Traiter des vidéos frame par frame pour repérer plusieurs véhicules.
- Extraire le texte des plaques détectées.
- Prendre en charge les plaques contenant des caractères latins et arabes.
- Afficher les résultats dans une interface web simple et moderne.
- Compter les plaques uniques détectées dans une vidéo.
- Sauvegarder les vidéos traitées avec les détections intégrées.

---

## 🧠 Technologies utilisées

| Catégorie | Technologie |
|---|---|
| Langage | Python 3.10+ |
| Détection d’objets | YOLOv8 / Ultralytics |
| OCR | EasyOCR |
| Traitement d’image | OpenCV |
| Deep Learning | PyTorch |
| Interface utilisateur | Streamlit |
| Annotation dataset | MakeSense.ai |
| Entraînement | Google Colab GPU Tesla T4 |
| Gestion du code | Git / GitHub |

---

## 🏗️ Architecture générale

Le système suit un pipeline complet de traitement d’image :

```text
Image ou vidéo d’entrée
        │
        ▼
Prétraitement de l’image
        │
        ▼
Détection de la plaque avec YOLOv8
        │
        ▼
Extraction / Crop de la zone de la plaque
        │
        ▼
Prétraitement OCR
- niveaux de gris
- amélioration du contraste
- seuillage / binarisation
        │
        ▼
Reconnaissance du texte avec EasyOCR
        │
        ▼
Affichage du résultat final
```

---

## ⚙️ Fonctionnalités principales

### 1. Détection sur image

L’utilisateur peut importer une image au format `JPG`, `JPEG` ou `PNG`.  
Le système détecte la plaque, l’encadre avec une boîte de délimitation, extrait la zone utile et affiche le texte reconnu.

### 2. Détection sur vidéo

L’application peut analyser une vidéo au format `MP4`, `AVI` ou `MOV`.  
Chaque frame est traitée par le modèle YOLOv8 afin de détecter les plaques présentes dans la scène.

### 3. Reconnaissance OCR

Après la détection, la plaque est envoyée vers EasyOCR afin d’extraire le contenu alphanumérique.  
Le système peut être configuré pour reconnaître :

- les chiffres ;
- les lettres latines ;
- certains caractères arabes utilisés dans les plaques.

### 4. Comptage intelligent

Pour les vidéos, le système intègre une logique de comptage permettant d’éviter les doublons.  
Chaque plaque détectée peut être associée à un identifiant afin de compter les véhicules uniques.

### 5. Sauvegarde des résultats

Les images ou vidéos traitées peuvent être sauvegardées avec les boîtes de détection et les résultats affichés.

---

## 📊 Performances du modèle

Les résultats obtenus lors de l’entraînement et de l’évaluation du modèle YOLOv8 montrent une bonne stabilité.

| Métrique | Résultat |
|---|---:|
| Précision | 98 % |
| Recall | 97 % |
| mAP50 | 0.98 |
| mAP50-95 | 0.725 |
| Nombre d’époques | 50 |
| Batch size | 16 |

Ces résultats montrent que le modèle détecte la majorité des plaques présentes dans les images tout en limitant les fausses détections.

---

## 📁 Structure recommandée du projet

```text
Reconnaissance-automatique-de-plaques-d-immatriculation/
│
├── app.py                         # Application Streamlit principale
├── requirements.txt               # Dépendances Python
├── README.md                      # Documentation du projet
│
├── models/
│   └── best.pt                    # Modèle YOLOv8 entraîné
│
├── data/
│   ├── images/                    # Images du dataset
│   ├── labels/                    # Annotations YOLO
│   └── data.yaml                  # Configuration du dataset YOLO
│
├── outputs/
│   ├── images/                    # Images traitées
│   └── videos/                    # Vidéos traitées
│
├── assets/
│   └── screenshots/               # Captures d’écran de l’interface
│
├── notebooks/
│   └── training_yolov8.ipynb      # Notebook d’entraînement Colab
│
└── utils/
    ├── detection.py               # Fonctions de détection YOLO
    ├── ocr.py                     # Fonctions EasyOCR
    └── preprocessing.py           # Prétraitement des images
```

> Remarque : adaptez cette structure selon les fichiers réels présents dans votre dépôt.

---

## 🚀 Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/VOTRE-USERNAME/Reconnaissance-automatique-de-plaques-d-immatriculation.git
cd Reconnaissance-automatique-de-plaques-d-immatriculation
```

### 2. Créer un environnement virtuel

#### Sur macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Sur Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📦 Exemple de fichier `requirements.txt`

```txt
streamlit
ultralytics
easyocr
opencv-python
numpy
pillow
torch
torchvision
pandas
matplotlib
tqdm
```

> Si vous utilisez un Mac avec puce Apple Silicon ou une version spécifique de PyTorch, installez PyTorch depuis la documentation officielle adaptée à votre machine.

---

## ▶️ Lancement de l’application

Lancer l’interface Streamlit :

```bash
streamlit run app.py
```

Ensuite, ouvrir l’adresse affichée dans le terminal, généralement :

```text
http://localhost:8501
```

---

## 🖼️ Utilisation avec une image

1. Ouvrir l’application Streamlit.
2. Choisir le mode **Image**.
3. Importer une image contenant un véhicule.
4. Lancer l’analyse.
5. Visualiser :
   - l’image originale ;
   - la plaque détectée ;
   - le texte extrait par EasyOCR ;
   - le score de confiance.

---

## 🎥 Utilisation avec une vidéo

1. Choisir le mode **Vidéo**.
2. Importer une vidéo au format `MP4`, `AVI` ou `MOV`.
3. Lancer le traitement.
4. Le système analyse les frames une par une.
5. Les plaques détectées sont encadrées.
6. Le compteur affiche le nombre de plaques uniques.
7. La vidéo traitée peut être exportée dans le dossier `outputs/videos/`.

---

## 🧪 Entraînement du modèle YOLOv8

Le modèle YOLOv8 a été entraîné par fine-tuning à partir d’un modèle pré-entraîné.

### Exemple de commande d’entraînement

```bash
yolo detect train \
  model=yolov8n.pt \
  data=data/data.yaml \
  epochs=50 \
  imgsz=640 \
  batch=16
```

### Exemple de fichier `data.yaml`

```yaml
path: ./data
train: images/train
val: images/val

names:
  0: plate
```

Après l’entraînement, le meilleur modèle est généralement disponible dans :

```text
runs/detect/train/weights/best.pt
```

Copiez ensuite ce fichier dans le dossier :

```text
models/best.pt
```

---

## 🔍 Exemple d’inférence YOLOv8

```python
from ultralytics import YOLO

model = YOLO("models/best.pt")
results = model("test_image.jpg", conf=0.5)

for result in results:
    result.show()
```

---

## 🔤 Exemple d’utilisation EasyOCR

```python
import easyocr

reader = easyocr.Reader(["ar", "en"], gpu=True)

result = reader.readtext(
    "plate_crop.jpg",
    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)

print(result)
```

---

## 🧩 Pipeline détaillé

### Étape 1 : Prétraitement

L’image est redimensionnée et normalisée pour être compatible avec YOLOv8.

### Étape 2 : Détection

YOLOv8 détecte la plaque et retourne les coordonnées de la boîte englobante.

### Étape 3 : Extraction de la plaque

La zone de la plaque est découpée à partir de l’image originale.

### Étape 4 : Prétraitement OCR

La plaque extraite subit plusieurs transformations :

- conversion en niveaux de gris ;
- amélioration du contraste ;
- seuillage ;
- redimensionnement.

### Étape 5 : OCR

EasyOCR lit les caractères présents dans la plaque et retourne le texte final.

---

## 📸 Captures d’écran

Ajoutez vos captures dans le dossier `assets/screenshots/`, puis remplacez les chemins ci-dessous.

### Interface principale

```md
![Interface principale](assets/screenshots/interface-principale.png)
```

### Détection sur image

```md
![Détection image](assets/screenshots/detection-image.png)
```

### Analyse vidéo

```md
![Analyse vidéo](assets/screenshots/analyse-video.png)
```

---

## ✅ Résultats de test

### Test sur image statique

- Entrée : image d’un véhicule.
- Action : détection de la plaque avec YOLOv8.
- Résultat : extraction du texte avec EasyOCR.
- Exemple de sortie : `0269 LKL`.

### Test sur trafic dense

- Entrée : vidéo contenant plusieurs véhicules.
- Action : détection frame par frame.
- Résultat : comptage des plaques uniques.
- Exemple observé : `7 plaques détectées`.

---

## ⚠️ Limites du projet

Malgré les bons résultats, certaines limites peuvent apparaître :

- plaques floues ou très petites ;
- mauvaise luminosité ;
- plaques partiellement cachées ;
- angles extrêmes ;
- reflets sur la plaque ;
- vitesse élevée du véhicule dans la vidéo ;
- erreurs OCR sur certains caractères proches visuellement.

---

## 🔮 Améliorations futures

- Ajouter un système de tracking plus avancé avec ByteTrack ou DeepSORT.
- Améliorer la correction automatique du texte OCR.
- Ajouter une base de données pour sauvegarder les plaques détectées.
- Ajouter un historique complet des détections.
- Déployer l’application sur Docker.
- Créer une API REST avec FastAPI ou Flask.
- Optimiser le modèle pour une exécution en temps réel sur caméra.
- Exporter les résultats en CSV ou Excel.
- Ajouter une authentification utilisateur pour un usage professionnel.

---

## 🐳 Option : exécution avec Docker

Créer un fichier `Dockerfile` :

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

Construire l’image :

```bash
docker build -t plate-ai-pro .
```

Lancer le conteneur :

```bash
docker run -p 8501:8501 plate-ai-pro
```

---

## 🛡️ Fichiers à ne pas pousser sur GitHub

Ajoutez un fichier `.gitignore` :

```gitignore
.venv/
__pycache__/
*.pyc

runs/
outputs/
*.mp4
*.avi
*.mov

data/images/
data/labels/
datasets/

*.pt
*.onnx
*.engine

.DS_Store
.env
```

> Les datasets volumineux et les poids du modèle peuvent être stockés séparément avec Git LFS, Google Drive ou Kaggle.

---

## 👥 Auteurs

Projet réalisé par :

- **Lahbichi Alae**
---

## 📚 Références

- YOLOv8 — Ultralytics
- EasyOCR
- OpenCV
- PyTorch
- Streamlit
- Kaggle datasets pour la détection de plaques d’immatriculation
- MakeSense.ai pour l’annotation des images

---

## 📄 Licence

Ce projet est réalisé dans un cadre académique.  
Vous pouvez ajouter une licence selon l’usage souhaité :

- `MIT License` pour un projet open source ;
- `Apache 2.0` pour un usage plus structuré ;
- licence privée si le projet ne doit pas être réutilisé.

---

## 📌 Résumé rapide

```text
PLATE AI PRO est une application ALPR basée sur YOLOv8 et EasyOCR.
Elle détecte les plaques d’immatriculation dans des images et vidéos,
extrait le texte automatiquement, affiche les résultats dans Streamlit
et permet le comptage intelligent des plaques détectées.
```
