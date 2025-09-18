import cv2
import os
import numpy as np
from pathlib import Path
import json

DATA_DIR = Path("data/dataset")
MODEL_DIR = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
model_path = MODEL_DIR / "face_model.yml"

def load_dataset(data_dir):
    faces = []
    labels = []
    label_map = {}
    current_label = 0

    for person_dir in sorted(data_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        label_name = person_dir.name
        label_map[current_label] = label_name
        for img_path in person_dir.glob("*.jpg"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(img)
            labels.append(current_label)
        current_label += 1

    return faces, np.array(labels), label_map

faces, labels, label_map = load_dataset(DATA_DIR)
if len(faces) == 0:
    raise SystemExit("[!] Nenhuma imagem encontrada em data/dataset. Rode capture_faces.py primeiro.")

recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)

print("[i] Treinando modelo...")
recognizer.train(faces, labels)
recognizer.write(str(model_path))

with open(MODEL_DIR / "labels.json", "w") as f:
    json.dump(label_map, f)

print(f"[i] Modelo salvo em {model_path}")
print(f"[i] Label map salvo em {MODEL_DIR / 'labels.json'}")
