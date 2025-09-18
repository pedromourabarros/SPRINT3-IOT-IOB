import cv2
import os
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Capturar imagens para dataset facial")
parser.add_argument("--id", required=True, help="ID da pessoa (ex: joao)")
parser.add_argument("--n", type=int, default=50, help="Número de imagens a capturar")
parser.add_argument("--out", default="data/dataset", help="Pasta de saída")
parser.add_argument("--cascade", default="haarcascade_frontalface_default.xml", help="Arquivo Haarcascade")
args = parser.parse_args()

cascade_path = args.cascade
if not Path(cascade_path).exists():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(cascade_path)
cap = cv2.VideoCapture(0)

out_dir = Path(args.out) / f"user_{args.id}"
out_dir.mkdir(parents=True, exist_ok=True)

count = 0
print(f"[i] Abrindo câmera. Pressione 'q' para sair. Salvando em {out_dir}")
while count < args.n:
    ret, frame = cap.read()
    if not ret:
        print("[!] Falha ao capturar frame da câmera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50,50))

    for (x,y,w,h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_img, (200,200))
        file_path = out_dir / f"img_{count:03d}.jpg"
        cv2.imwrite(str(file_path), face_resized)
        count += 1

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, f"Captura {count}/{args.n}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)

    cv2.imshow("Captura - Press q para sair", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"[i] Captura finalizada. Capturadas: {count}")
cap.release()
cv2.destroyAllWindows()
