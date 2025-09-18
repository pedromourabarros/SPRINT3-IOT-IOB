import cv2
import json
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="data/models/face_model.yml")
parser.add_argument("--labels", default="data/models/labels.json")
parser.add_argument("--threshold", type=float, default=60.0)
args = parser.parse_args()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

if not Path(args.model).exists():
    raise SystemExit("[!] Rode train_model.py primeiro.")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(str(args.model))

with open(args.labels, "r") as f:
    label_map = json.load(f)
inv_map = {int(k): v for k, v in label_map.items()}

cap = cv2.VideoCapture(0)
print("[i] Pressione 'q' para encerrar.")
while True:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50,50))
    for (x,y,w,h) in faces:
        face = cv2.resize(gray[y:y+h,x:x+w], (200,200))
        label_id, confidence = recognizer.predict(face)
        text = "Unknown"
        if confidence < args.threshold:
            text = inv_map.get(label_id, "Unknown")
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, f"{text} ({confidence:.1f})", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imshow("Reconhecimento Facial", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release()
cv2.destroyAllWindows()
